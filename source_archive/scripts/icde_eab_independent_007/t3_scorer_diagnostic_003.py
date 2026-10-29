"""Pinned ColPali pilot. Only predeclared Dev queries, unlabeled corpus pages."""
import common as C
import os, sys, time, json, io, shutil, subprocess, resource, ast, collections, math
from pathlib import Path
import numpy as np
gpu=int(sys.argv[1]);domain=sys.argv[2];proc=int(sys.argv[3])
stage=f'T3/scorer_diagnostic_{domain}_003';C.start(stage)
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
checks=[]
for _ in range(3):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True);a,b=raw.strip().split(',');ok=int(a.split()[0])<128 and int(b.split()[0])==0;checks.append({'time_bj':C.stamp(),'raw':raw,'idle':ok})
    if not ok:C.write(C.A/'three_idle_checks.json',checks);raise RuntimeError('BUSY_GPU_NO_CUDA')
    time.sleep(1)
C.write(C.A/'three_idle_checks.json',checks);sys.path.insert(0,str(C.ROOT/'src'))
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
from mdrouter_v5.graded_metrics import ndcg,b1l,stable_ranking
guard=ResourceGuard(gpu,C.A/'resource_guard.json',period=.1)
try:
    import torch,pyarrow.parquet as pq
    from PIL import Image
    from colpali_engine.models import ColPali,ColPaliProcessor
    from aaber.common.text import tokenize
    torch.set_num_threads(1);torch.manual_seed(C.SEED);torch.backends.cuda.matmul.allow_tf32=False
    prep=C.load(C.ROOT/'artifacts/icde_eab_main_006/X1/preparation.json');base=Path(prep['model_path']['vidore/colpaligemma-3b-pt-448-base']);adapter=Path(prep['model_path']['vidore/colpali-v1.3'])
    # Private configuration overlay pins an already-cached base; original cache remains read-only.
    from colpali_loader_003 import prepare_overlay,validate_model
    overlay,mapping=prepare_overlay(C,C.A/'adapter_overlay')
    t=time.perf_counter();model=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='eager').eval();processor=ColPaliProcessor.from_pretrained(str(adapter),local_files_only=True);torch.cuda.synchronize();setup={'model_load_wall_s':time.perf_counter()-t,'active_adapters':model.active_adapters(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'processor_image_size':{'height':processor.image_processor.size.height,'width':processor.image_processor.size.width}}
    assert model.active_adapters(),'trained adapter not active';C.write(C.A/'adapter_loaded_tensors.json',validate_model(model,overlay,mapping))
    primary={'vidore_v3_hr':'hr_primary_001','vidore_v3_industrial':'industrial_primary_003'}[domain]
    pages=C.load(C.ROOT/'cache/indexes'/primary/'pages.json','UNLABELLED_CORPUS_TEXT');pages.sort(key=lambda r:int(r['corpus_id']));ids=[int(r['corpus_id']) for r in pages];pos={v:i for i,v in enumerate(ids)}
    nodes=[x for x in ast.parse(C.load('scripts/cross_domain_four_plan_001.py')).body if isinstance(x,ast.ClassDef) and x.name=='BM25'];env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'pinned_BM25','exec'),env);t=time.perf_counter();bm=env['BM25'](pages);setup['BM25_setup_wall_s']=time.perf_counter()-t
    inventory=C.load('manifests/primary_datasets_inventory.json');ds=next(x for x in inventory['datasets'] if x['repo'].split('/')[-1]==domain);corpus=C.ROOT/'data/primary'/domain/ds['revision']/'corpus';index=C.ROOT/'cache/indexes'/f'{C.RUN}_colpali_{domain}_002'
    if False:
        index.mkdir(exist_ok=False);(index/'visual_parts').mkdir();t=time.perf_counter();ct=time.process_time();done=[]
        with torch.inference_mode():
            for shard in sorted(corpus.glob('*.parquet')):
                expected=next(x['lfs_sha256'] for x in ds['files'] if x['path']=='corpus/'+shard.name);assert C.sha(shard)==expected
                for block in pq.ParquetFile(shard).iter_batches(batch_size=2,columns=['corpus_id','image']):
                    guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30;rows=block.to_pylist();imgs=[Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in rows];inp=processor.process_images(imgs).to('cuda:0');em=model(**inp);valid=inp['attention_mask'].bool();ee=[x[m].detach().to(dtype=torch.float16).cpu() for x,m in zip(em,valid)];assert all(torch.isfinite(x).all() for x in ee);rid=[int(r['corpus_id']) for r in rows];torch.save({'ids':rid,'embeddings':ee},index/'visual_parts'/f'{rid[0]:06d}.pt');done+=rid
                    if len(done)%100<2:print(json.dumps({'stage':stage,'pages':len(done),'total':len(ids),'wall_s':time.perf_counter()-t,'time_bj':C.stamp()}),flush=True)
        assert set(done)==set(ids)
        C.write(index/'index_manifest.json',{'corpus_pages':len(ids),'setup_wall_s':time.perf_counter()-t,'setup_CPU_s':time.process_time()-ct,'setup_GPU_activity_s':None,'setup_GPU_scope':'unprofiled index build; not imputed from wall','peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_host_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'files':{str(p.relative_to(index)):C.sha(p) for p in index.rglob('*.pt')},'model_preparation_SHA256':C.sha(C.ROOT/'artifacts/icde_eab_main_006/X1/preparation.json'),'processor':setup['processor_image_size']})
    im=C.load(index/'index_manifest.json','UNLABELLED_VISUAL_INDEX');vv={};t=time.perf_counter()
    for part in sorted((index/'visual_parts').glob('*.pt')):
        guard.check();assert C.sha(part)==im['files'][str(part.relative_to(index))];z=torch.load(part,weights_only=True);vv.update(zip(z['ids'],z['embeddings']))
    passage=torch.nn.utils.rnn.pad_sequence([vv[i].float() for i in ids],batch_first=True).cuda();torch.cuda.synchronize();setup['index_residency_wall_s']=time.perf_counter()-t;setup['index_manifest_SHA256']=C.sha(index/'index_manifest.json');C.write(C.A/'setup.json',setup)
    allow=C.load(C.REPORT/'T0/colpali_pilot_allowlist.csv');sel=[int(r['query_id']) for r in allow if r['domain']==domain];queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json');queries={int(q['query_id']):q['query'] for q in queries};labels=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_qrels.json','AUTHORIZED_DEV_QRELS');ql=collections.defaultdict(dict)
    for r in labels:ql[int(r['query_id'])][int(r['corpus_id'])]=int(r['score'])
    def score(q,indices):
        values=[]
        for s in range(0,len(indices),64):
            ix=indices[s:s+64];block=passage.index_select(0,torch.tensor(ix,device='cuda:0'));v=torch.matmul(q.unsqueeze(0),block.transpose(1,2)).max(2).values.clamp_min(0).sum(1);values+=v.cpu().tolist()
        order=stable_ranking([ids[i] for i in indices],values);return [indices[j] for j in order],values
    def execute(plan,text):
        if plan=='Full-Visual':cand=list(range(len(ids)))
        else:order,_=bm.search(text);cand=order[:int(plan[6:])]
        if not cand:return []
        qe=model(**processor.process_queries([text]).to('cuda:0'))[0].float();ranking,_=score(qe,cand);return [ids[i] for i in ranking[:10]]

    import inspect
    C.prose(C.A/'official_score_source.txt',inspect.getsource(processor.score_multi_vector))
    diag=[]
    with torch.inference_mode():
        for qid in sel:
            qe=model(**processor.process_queries([queries[qid]]).to('cuda:0'))[0].float();sample=list(range(12));ours,vals=score(qe,sample);official=processor.score_multi_vector([qe],[vv[ids[i]].float() for i in sample],device='cuda:0')[0].cpu().numpy()
            for j,i in enumerate(sample):
                v=vv[ids[i]].float().cuda();maxes=(qe@v.T).max(1).values;unclamped=maxes.sum().item();clamped=maxes.clamp_min(0).sum().item();diag.append({'query_id':qid,'page_id':ids[i],'ours':vals[j],'official':float(official[j]),'error':vals[j]-float(official[j]),'direct_unclamped':unclamped,'direct_clamped':clamped,'negative_token_maxima':int((maxes<0).sum()),'query_tokens':len(qe),'page_tokens':len(v),'global_padded_tokens':passage.shape[1]})
    C.table(C.R/'score_diagnostic.csv',diag);C.write(C.A/'diagnostic_summary.json',{'max_abs_error':max(abs(x['error']) for x in diag),'negative_max_token_entries':sum(x['negative_token_maxima'] for x in diag),'time_bj':C.stamp(),'status':'DIAGNOSTIC_NOT_QUALITY_EVALUATION'})
    print(json.dumps({'rows':len(diag),'max_abs_error':max(abs(x['error']) for x in diag),'negative_token_maxima':sum(x['negative_token_maxima'] for x in diag),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
