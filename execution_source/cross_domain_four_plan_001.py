"""Real four-plan engineering pilot. Only frozen dev queries/qrels are read."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from colpali_engine.models import ColQwen2, ColQwen2Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aaber.common.text import tokenize
from mdrouter_v5.graded_metrics import ndcg, b1l, stable_ranking
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call,summarize_trace

CONFIG_PATH = ROOT / os.environ['MDROUTER_CONFIG']
CFG = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
MODELS = json.loads((ROOT / 'manifests/hr_models_verified.json').read_text())
SPLIT = ROOT / 'artifacts/splits' / CFG['split_id']
SOURCE = next((ROOT / 'data/primary' / CFG['dataset']).iterdir())
INDEX = ROOT / 'cache/indexes' / CFG['index_id']
OUT = ROOT / 'artifacts/phase1' / CFG['run_id']
RESOURCE = None

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def guard():
    if RESOURCE is not None:RESOURCE.check()
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    status = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    print('GPU BEFORE LOAD', status, flush=True)
    used = int(status.split(',')[2].strip().split()[0])
    if used >= 128:
        raise RuntimeError('Chosen GPU is occupied; no model loaded')
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    np.random.seed(20261009)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return status

def load_visual():
    path = MODELS['visual']['local_path']
    processor = ColQwen2Processor.from_pretrained(path, local_files_only=True)
    mapping = {r'^base_model\.model\.custom_text_proj': 'custom_text_proj',
               r'^model\.embed_tokens': 'language_model.embed_tokens',
               r'^model\.layers': 'language_model.layers', r'^model\.norm': 'language_model.norm'}
    model, info = ColQwen2.from_pretrained(path, dtype=torch.float16, device_map='cuda:0', key_mapping=mapping,
        local_files_only=True, output_loading_info=True, attn_implementation='sdpa')
    assert not any(info.values()), info
    return model.eval(), processor

def load_dense():
    path = MODELS['dense']['local_path']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    check = 'Tokenizer compatibility check.'
    assert tokenizer.encode(check) == [tokenizer.cls_token_id] + tokenizer.encode(check, add_special_tokens=False) + [tokenizer.sep_token_id]
    model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).to('cuda:0').eval()
    return model, tokenizer

def encode_dense(model, tokenizer, texts=None, token_ids=None):
    if texts is not None:
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    else:
        records = [{'input_ids': [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id],
                    'attention_mask': [1] * (len(ids) + 2)} for ids in token_ids]
        inputs = tokenizer.pad(records, padding=True, return_tensors='pt')
    output = model(**inputs.to('cuda:0')).last_hidden_state[:, 0].float()
    return torch.nn.functional.normalize(output, dim=1)

def build():
    status = guard()
    INDEX.mkdir(parents=True, exist_ok=True)
    assert not (INDEX / 'index_manifest.json').exists(), 'Completed index is immutable'
    t0 = time.perf_counter()
    pages = [r for p in sorted((SOURCE / 'corpus').glob('*.parquet'))
             for r in pq.read_table(p, columns=['corpus_id', 'doc_id', 'markdown', 'page_number_in_doc']).to_pylist()]
    pages.sort(key=lambda r: int(r['corpus_id']))
    assert len(pages) == CFG['expected_pages'] and len({r['corpus_id'] for r in pages}) == CFG['expected_pages']
    write(INDEX / 'pages.json', pages)
    dense_model, tokenizer = load_dense()
    chunks, chunk_pages, lengths = [], [], []
    for i, p in enumerate(pages):
        ids = tokenizer.encode(p['markdown'] or '', add_special_tokens=False)
        lengths.append(len(ids))
        for start in range(0, max(1, len(ids)), CFG['dense']['chunk_stride']):
            chunk = ids[start:start + CFG['dense']['chunk_tokens']]
            chunks.append(chunk)
            chunk_pages.append(i)
            if start + CFG['dense']['chunk_tokens'] >= len(ids):
                break
    print('DENSE CHUNKS', len(chunks), 'page max tokens', max(lengths), flush=True)
    embeddings = []
    dense_t = time.perf_counter()
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as dense_profile:
        for start in range(0, len(chunks), 16):
            if RESOURCE is not None:RESOURCE.check()
            emb = encode_dense(dense_model, tokenizer, token_ids=chunks[start:start+16])
            assert torch.isfinite(emb).all()
            embeddings.append(emb.cpu())
            if start % 160 == 0:
                print('DENSE INDEX', min(start+16, len(chunks)), '/', len(chunks), 'elapsed', round(time.perf_counter()-dense_t, 1), flush=True)
    dense_profile.export_chrome_trace(str(INDEX/'dense_build_gpu_trace.json'))
    dense_gpu=summarize_trace(INDEX/'dense_build_gpu_trace.json')
    dense = torch.cat(embeddings)
    torch.save({'embeddings': dense, 'chunk_pages': torch.tensor(chunk_pages, dtype=torch.long)}, INDEX / 'dense.pt')
    dense_seconds = time.perf_counter() - dense_t
    del dense_model, tokenizer, dense, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    visual_model, processor = load_visual()
    # Transformers 5 exposes the pinned resize budget through image_processor.size.
    assert processor.image_processor.size['longest_edge'] == 768 * 28 * 28
    visual_t = time.perf_counter()
    done = set()
    image_hashes = []
    for shard in sorted((SOURCE / 'corpus').glob('*.parquet')):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=CFG['visual']['batch_size'], columns=['corpus_id', 'image']):
            if RESOURCE is not None:RESOURCE.check()
            rows = batch.to_pylist()
            ids = [int(r['corpus_id']) for r in rows]
            path = INDEX / 'visual_parts' / f'{ids[0]:06d}.pt'
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                cached = torch.load(path, weights_only=True)
                assert cached['ids'] == ids
                done.update(ids)
                continue
            images = [Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in rows]
            for pid, image in zip(ids, images):
                image_hashes.append({'page_id': pid, 'decoded_pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            with torch.inference_mode():
                inputs = processor.process_images(images).to('cuda:0')
                embs = visual_model(**inputs)
                valid = inputs['attention_mask'].bool()
                embs = [e[m].detach().cpu() for e, m in zip(embs, valid)]
                assert all(bool(torch.isfinite(e).all()) for e in embs)
            torch.save({'ids': ids, 'embeddings': embs}, path)
            done.update(ids)
            if len(done) % 20 == 0 or len(done) == len(pages):
                print('VISUAL INDEX', len(done), '/', len(pages), 'elapsed', round(time.perf_counter()-visual_t, 1), flush=True)
    assert done == {int(p['corpus_id']) for p in pages}
    visual_seconds = time.perf_counter() - visual_t
    write(INDEX / 'image_exact_duplicate_audit.json', image_hashes)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'pages': len(pages), 'chunks': len(chunk_pages),
        'dense_seconds': dense_seconds, 'visual_seconds': visual_seconds, 'total_setup_seconds': time.perf_counter()-t0,
        'dense_gpu_accounting':dense_gpu,'visual_gpu_accounting':'NOT_RUN; shared reference/route index, wall setup recorded separately',
        'config_sha256': sha(CONFIG_PATH), 'model_manifest_sha256': sha(ROOT / 'manifests/hr_models_verified.json'),
        'frozen_backend_source': 'hr_pilot_004', 'code_sha256': sha(Path(__file__)),
        'source_corpus': {str(p.relative_to(ROOT)): sha(p) for p in sorted((SOURCE / 'corpus').glob('*.parquet'))},
        'gpu_before_load': status, 'markdown_token_counts': {'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths))},
        'files': {str(p.relative_to(INDEX)): sha(p) for p in sorted(INDEX.rglob('*')) if p.is_file()}}
    write(INDEX / 'index_manifest.json', manifest)
    print('INDEX COMPLETE', json.dumps({k: manifest[k] for k in ['pages', 'chunks', 'dense_seconds', 'visual_seconds']}), flush=True)

class BM25:
    def __init__(self, pages):
        self.pages = pages
        self.ids = [int(p['corpus_id']) for p in pages]
        self.inverted = defaultdict(list)
        lengths = []
        for i, p in enumerate(pages):
            counts = Counter(tokenize(p['markdown']))
            lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.inverted[term].append((i, freq))
        self.lengths = np.array(lengths)
        self.avg = max(float(self.lengths.mean()), 1e-9)
        self.contributions = {}
        k1, b = CFG['bm25']['k1'], CFG['bm25']['b']
        for term, entries in self.inverted.items():
            index = np.array([i for i, _ in entries], dtype=np.int64)
            frequency = np.array([f for _, f in entries], dtype=np.float64)
            idf = math.log(1 + (len(self.pages)-len(entries)+0.5)/(len(entries)+0.5))
            weight = idf*frequency*(k1+1)/(frequency+k1*(1-b+b*self.lengths[index]/self.avg))
            self.contributions[term] = (index, weight)
    def search(self, question, limit=None):
        scores = np.zeros(len(self.pages), dtype=np.float64)
        k1, b = CFG['bm25']['k1'], CFG['bm25']['b']
        for term in tokenize(question):
            contribution = self.contributions.get(term)
            if contribution is None:
                continue
            index, weight = contribution
            scores[index] += weight
        order = [i for i in stable_ranking(self.ids, scores) if scores[i] > 0]
        if limit is not None:
            order = order[:limit]
        return order, scores

def evaluate():
    status = guard()
    OUT.mkdir(parents=True, exist_ok=False)
    index_manifest = json.loads((INDEX / 'index_manifest.json').read_text())
    original = json.loads((ROOT / 'configs/hr_pilot.json').read_text())
    assert index_manifest['config_sha256'] == sha(CONFIG_PATH)
    assert index_manifest['model_manifest_sha256'] == sha(ROOT / 'manifests/hr_models_verified.json')
    assert CFG['dense'] == original['dense'] and {k:v for k,v in CFG['bm25'].items() if k!='arithmetic'} == original['bm25']
    assert {k:v for k,v in CFG['visual'].items() if k not in ['score','score_batch_size']} == {k:v for k,v in original['visual'].items() if k != 'score'}
    assert CFG['dataset'] in ['vidore_v3_finance_en','vidore_v3_industrial','vidore_v3_hr']
    assert CFG['split_id'] == 'query_generalization_002'
    pages = json.loads((INDEX / 'pages.json').read_text())
    ids = [int(p['corpus_id']) for p in pages]
    id_to_index = {pid: i for i, pid in enumerate(ids)}
    queries = json.loads((SPLIT / CFG['dataset'] / 'dev_queries.json').read_text())
    labels = json.loads((SPLIT / CFG['dataset'] / 'dev_qrels.json').read_text())
    assert len(queries) == CFG['query_count'] and len({q['query_id'] for q in queries}) == CFG['query_count']
    qrels = defaultdict(dict)
    for row in labels:
        assert int(row['query_id']) in {int(q['query_id']) for q in queries}, 'Non-development qrel in execution entrypoint'
        qrels[int(row['query_id'])][int(row['corpus_id'])] = int(row['score'])
    bm25 = BM25(pages)
    load_t = time.perf_counter()
    dense_model, tokenizer = load_dense()
    dense = torch.load(INDEX / 'dense.pt', weights_only=True)
    dense_emb = dense['embeddings'].cuda()
    chunk_pages = dense['chunk_pages'].cuda()
    visual_model, processor = load_visual()
    visual_pages = {}
    for path in sorted((INDEX / 'visual_parts').glob('*.pt')):
        part = torch.load(path, weights_only=True)
        visual_pages.update(zip(part['ids'], part['embeddings']))
    # Resident FP32 passage embeddings: scoring arithmetic is identical for each page in both paths.
    passage = torch.nn.utils.rnn.pad_sequence([visual_pages[pid].float() for pid in ids], batch_first=True).cuda()
    assert len(passage) == CFG['expected_pages']
    model_index_load_seconds = time.perf_counter()-load_t
    def query_visual(question):
        return visual_model(**processor.process_queries([question]).to('cuda:0'))[0].float()
    def score_visual(question, indices):
        if not indices:
            return [], {}, None
        q = query_visual(question)
        batch_size = CFG['visual']['score_batch_size']
        result = []
        for start in range(0, len(indices), batch_size):
            subset = indices[start:start+batch_size]
            block = passage.index_select(0, torch.tensor(subset, device='cuda:0'))
            if len(subset) < batch_size:
                block = torch.nn.functional.pad(block, (0,0,0,0,0,batch_size-len(subset)))
            value = torch.matmul(q.unsqueeze(0).expand(batch_size,-1,-1), block.transpose(1,2)).max(dim=2).values.clamp_min(0).sum(dim=1)
            result.append(value[:len(subset)])
        vals = torch.cat(result).cpu().tolist()
        ordered = stable_ranking([ids[i] for i in indices], vals)
        return [indices[j] for j in ordered], {ids[i]: float(v) for i, v in zip(indices, vals)}, q
    def execute(plan, question):
        if plan == 'T-BM25':
            order, scores = bm25.search(question)
            return order, {ids[i]: float(scores[i]) for i in order}
        if plan == 'T-Dense':
            q = encode_dense(dense_model, tokenizer, texts=[CFG['dense']['query_prefix'] + question])[0]
            chunk_scores = dense_emb @ q
            scores = torch.full((len(ids),), -float('inf'), device='cuda:0')
            scores.scatter_reduce_(0, chunk_pages, chunk_scores, reduce='amax', include_self=True)
            vals = scores.cpu().tolist()
            return stable_ranking(ids, vals), dict(zip(ids, vals))
        if plan == 'T-to-V-Rerank@50':
            candidates, _ = bm25.search(question, CFG['rerank']['k'])
            order, scores, _ = score_visual(question, candidates)
            return order, scores
        order, scores, _ = score_visual(question, list(range(len(ids))))
        return order, scores
    costs, outcomes, rankings, diagnostics, validations = [], [], [], [], []
    # Warm each path on a developer query; these measurements do not enter profiles.
    with torch.inference_mode():
        for plan in CFG['plans']:
            execute(plan, queries[0]['query'])
        for qi, query in enumerate(queries):
            if RESOURCE is not None:RESOURCE.check()
            qid, question = int(query['query_id']), query['query']
            first = {}
            for rep in range(CFG['repeats']):
                plans = CFG['plans'].copy()
                random.Random(20261009 + qi*17 + rep).shuffle(plans)
                for position, plan in enumerate(plans):
                    torch.cuda.synchronize()
                    start_cpu, start = time.process_time(), time.perf_counter()
                    event1, event2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    event1.record()
                    order, scores = execute(plan, question)
                    event2.record()
                    torch.cuda.synchronize()
                    wall = (time.perf_counter()-start)*1000
                    cpu = (time.process_time()-start_cpu)*1000
                    costs.append({'query_id': qid, 'dataset': CFG['dataset'], 'split': 'dev', 'plan_id': plan,
                        'repeat': rep, 'position': position, 'wall_ms': wall, 'process_cpu_ms': cpu,
                        'cuda_elapsed_span_ms': event1.elapsed_time(event2) if plan != 'T-BM25' else None,
                        'energy_joules': None, 'cache': 'resident', 'run_id': CFG['run_id']})
                    page_order = [ids[i] for i in order]
                    if rep == 0:
                        first[plan] = (page_order, scores)
                    else:
                        assert page_order == first[plan][0], (qid, plan, 'repeat ranking changed')
            ref = first['V-Multi'][0]
            uref = ndcg(ref, qrels[qid])
            for plan in CFG['plans']:
                ranking, scores = first[plan]
                u = ndcg(ranking, qrels[qid])
                true_r = max(0, uref-u)
                bound = b1l(ref, ranking, {pid: qrels[qid].get(pid, 0) for pid in ref[:10]})
                assert -1e-12 <= true_r <= bound + 1e-12 <= 1+1e-12
                record = {'query_id': qid, 'dataset': CFG['dataset'], 'split': 'dev', 'plan_id': plan,
                    'U': u, 'U_ref': uref, 'r': true_r, 'r_bar': bound, 'slack': bound-true_r,
                    'signed_gap': uref-u, 'positive_pages': len(qrels[qid]), 'query_generator': query['query_generator'],
                    'status': 'ok', 'ranking_ref': f'rankings.jsonl#{qid}:{plan}'}
                outcomes.append(record)
                rankings.append({'query_id': qid, 'plan_id': plan, 'page_ids': ranking, 'scores': [scores[p] for p in ranking]})
            full_order, lexical_scores = bm25.search(question)
            candidates = [ids[i] for i in full_order[:50]]
            score_difference = max((abs(first['V-Multi'][1][p]-first['T-to-V-Rerank@50'][1][p]) for p in candidates), default=0)
            assert score_difference == 0, (qid, 'full/subset scoring mismatch', score_difference)
            vals = {p: first['V-Multi'][1][p] for p in [ids[i] for i in full_order[:100]]}
            for k in [10, 50, 100]:
                selected = [ids[i] for i in full_order[:k]]
                ranking = sorted(selected, key=lambda p: (-vals[p], p))
                diagnostics.append({'query_id': qid, 'k': k, 'candidate_count': len(selected),
                    'candidate_recall': len(set(selected)&qrels[qid].keys())/len(qrels[qid]),
                    'no_relevant_candidate': not bool(set(selected)&qrels[qid].keys()),
                    'rerank_U': ndcg(ranking, qrels[qid]), 'cost_measured': k == 50,
                    'posterior_empty_mass': None, 'posterior_status': 'not trained; no zero substituted'})
            validations.append({'query_id': qid, 'full_subset_score_max_abs_difference': score_difference,
                'repeat_rankings_equal': True, 'single_relevant': len(qrels[qid]) == 1,
                'query_token_count': len(tokenizer.encode(CFG['dense']['query_prefix']+question))})
            print('QUERY', qi+1, '/',len(queries), 'id', qid, 'U', {p: round(ndcg(first[p][0], qrels[qid]), 4) for p in CFG['plans']},
                'score_diff', score_difference, flush=True)
            with (OUT / 'progress.jsonl').open('a') as f:
                f.write(json.dumps({'query_done': qi+1, 'query_id': qid, 'at_utc': datetime.now(timezone.utc).isoformat()})+'\n')
            # Completed query diagnostics survive an interrupted run; never impute unexecuted plans.
            with (OUT/'partial_outcomes.jsonl').open('a') as f:
                for row in outcomes[-len(CFG['plans']):]:f.write(json.dumps(row)+'\n')
    with (OUT / 'rankings.jsonl').open('w') as f:
        for r in rankings:
            f.write(json.dumps(r)+'\n')
    for name, rows in [('query_plan_outcomes', outcomes), ('cost_profiles', costs), ('probe_diagnostics', diagnostics)]:
        pq.write_table(pa.Table.from_pylist(rows), OUT / (name+'.parquet'))
        write(OUT / (name+'.json'), rows)
    write(OUT / 'validations.json', validations)
    gpu_costs=[]
    expected={(r['query_id'],r['plan_id']):r['page_ids'] for r in rankings}
    with torch.inference_mode():
        for qi,query in enumerate(queries):
            qid,question=int(query['query_id']),query['query']
            for rep in range(CFG['repeats']):
                plans=CFG['plans'].copy();random.Random(20261009+qi*17+rep).shuffle(plans)
                for position,plan in enumerate(plans):
                    if RESOURCE is not None:RESOURCE.check()
                    trace=OUT/'gpu_traces'/f'{qid}_{rep}_{plan}.json'
                    result,account=profile_call(lambda:execute(plan,question),trace,require_kernels=plan!='T-BM25')
                    assert [ids[i] for i in result[0]]==expected[qid,plan],(qid,plan,'profiled ranking changed')
                    if plan=='T-BM25':assert account['activity_events']==0,'Unexpected GPU activity for CPU-only BM25'
                    row={'query_id':qid,'dataset':CFG['dataset'],'split':'dev','plan_id':plan,'repeat':rep,'position':position,
                        'run_id':CFG['run_id'],'gpu_trace':str(trace.relative_to(ROOT)),'gpu_trace_sha256':sha(trace),
                        'ranking_matches_unprofiled':True,**account}
                    gpu_costs.append(row)
                    with (OUT/'partial_gpu_costs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('GPU COST QUERY',qi+1,'/',len(queries),'id',qid,flush=True)
    pq.write_table(pa.Table.from_pylist(gpu_costs),OUT/'cost_profiles_gpu.parquet')
    write(OUT/'cost_profiles_gpu.json',gpu_costs)
    # Empty lexical query branch explicitly tested without making a gold judgment.
    empty, _ = bm25.search('')
    assert empty == [] and execute('T-to-V-Rerank@50', '') == ([], {})
    write(OUT / 'execution_manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': CFG['dataset'], 'dev_queries': len(queries), 'plan_outcomes': len(outcomes), 'cost_rows': len(costs),'gpu_cost_rows':len(gpu_costs),
        'repeats_are_independent_queries': False, 'final_test_evaluated': False, 'B_L_labels_used': 0,
        'empty_candidate_test': 'returned empty; no fallback', 'model_index_load_seconds': model_index_load_seconds,
        'gpu_before_load': status, 'index_manifest_sha256': sha(INDEX / 'index_manifest.json'),
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [CONFIG_PATH, ROOT / 'configs/hr_pilot.json', ROOT / 'manifests/hr_models_verified.json', SPLIT / 'split_manifest.json', SPLIT / CFG['dataset'] / 'dev_queries.json', SPLIT / CFG['dataset'] / 'dev_qrels.json']},
        'code': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/mdrouter_v5/graded_metrics.py',ROOT/'src/mdrouter_v5/gpu_accounting.py', ROOT / 'src/aaber/common/text.py']},
        'limits': ['CUDA elapsed span is not isolated active kernel time', 'no serving queue/arrival/batching measurement', 'energy missing', 'semantic/image near-duplicate review incomplete']})
    print(f'PILOT COMPLETE queries={len(queries)} outcomes={len(outcomes)} costs={len(costs)} GPUcosts={len(gpu_costs)}; final_test_evaluated=false', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'evaluate'])
    stage = parser.parse_args().stage
    destination=ROOT/'artifacts/cross_domain_k1_001/resources'/f'{CFG["run_id"]}_{stage}.json'
    RESOURCE=ResourceGuard(int(os.environ['CUDA_VISIBLE_DEVICES']),destination)
    try:build() if stage == 'build' else evaluate()
    finally:RESOURCE.close()
