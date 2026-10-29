"""Frozen physical plans, new cost cohort only. Never loads qrels or quality labels."""
import time
EARLY=time.perf_counter()
import common as C
import ast,collections,io,json,math,os,resource,shutil,subprocess,sys
from pathlib import Path
import numpy as np
model_name=sys.argv[1];proc=int(sys.argv[2]);gpu=int(sys.argv[3]);assert model_name in ['ColQwen2','ColPali'] and proc in range(4)
C.start(f'Y1/{model_name}_p{proc}_001');os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='1'
checks=[]
for _ in range(3):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();x=raw.split(',');ok=int(x[2].split()[0])<128 and int(x[3].split()[0])==0;checks.append({'time_bj':C.stamp(),'raw':raw,'idle':ok})
    if not ok:C.write(C.A/'three_idle_checks.json',checks);raise RuntimeError('BUSY_BEFORE_CUDA_STOP_OWN_WORKER')
    time.sleep(1)
C.write(C.A/'three_idle_checks.json',checks);operational_start=time.perf_counter();sys.path.insert(0,str(C.ROOT/'src'))
from mdrouter_v5.gpu_accounting import ResourceGuard
from mdrouter_v5.graded_metrics import stable_ranking
guard=ResourceGuard(gpu,C.A/'resource_guard.json',period=.1)
try:
    import torch,pyarrow.parquet as pq
    from PIL import Image
    from gpu_profile_001 import profile
    import sys
    sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_independent_007'));import engine_exact_004 as E
    torch.set_num_threads(1);torch.manual_seed(C.SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bio=C.load('artifacts/icde_eab_independent_007/T1/biomedical_frozen_protocol.json','FROZEN_BENCHMARK_DESIGN');design=C.load(C.ART/'Y0/frozen_protocol.json','009_COST_DESIGN')
    for p,h in bio['code_SHA256'].items():assert C.sha(C.ROOT/p)==h,'FROZEN_CODE_CHANGED'
    for kind in ['corpus','queries']:assert C.sha(C.ROOT/bio['official_inputs'][kind]['path'])==bio['official_inputs'][kind]['sha256']
    qt=C.ROOT/bio['official_inputs']['queries']['path'];queries={int(x['query-id']):x['query'] for x in pq.read_table(qt).to_pylist()};assert sorted(queries)==bio['query_ids']
    C.write(C.A/'read_audit.json',{'time_bj':C.stamp(),'allowed_query_scope':'public Biomedical official160 for timing only','query_SHA256':C.sha(qt),'qrels_read':False,'sealed_reads':0})
    pages=C.load('artifacts/icde_eab_independent_007/T1_preparation/ocr_001/pages.json','SHARED_ALREADY_MEASURED_OCR');pages.sort(key=lambda x:int(x['corpus_id']));ids=[int(x['corpus_id']) for x in pages];assert len(ids)==1016
    env=E.backend();records=[]
    def measure(name,fn,profiled=False):
        guard.check();torch.cuda.synchronize();st=time.perf_counter();ct=time.process_time()
        if profiled:out,g=profile(fn,C.A/(name+'_GPU.json'),validate=name=='model_load',require_kernels=False)
        else:out=fn();g=None
        torch.cuda.synchronize();records.append({'model':model_name,'process':proc,'component':name,'wall_s':time.perf_counter()-st,'CPU_s':time.process_time()-ct,'GPU_s':g['gpu_active_union_seconds'] if g else None,'cohort':'PROFILED' if profiled else 'UNPROFILED','GPU_scope':'actual activity union' if g else 'NOT_MEASURED_THIS_COHORT','peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});guard.check();return out
    st=time.perf_counter();ct=time.process_time();torch.cuda.init();torch.cuda.synchronize();records.append({'model':model_name,'process':proc,'component':'CUDA_context_init','wall_s':time.perf_counter()-st,'CPU_s':time.process_time()-ct,'GPU_s':None,'GPU_scope':'NOT_MEASURED_CONTEXT_CREATION','cohort':'UNPROFILED'})
    if model_name=='ColPali':
        from colpali_engine.models import ColPali,ColPaliProcessor
        from colpali_loader_003 import prepare_overlay,validate_model
        overlay,mapping=measure('adapter_overlay',lambda:prepare_overlay(C,C.A/'adapter_overlay'))
        prep=C.load('artifacts/icde_eab_main_006/X1/preparation.json');adapter=prep['model_path']['vidore/colpali-v1.3']
        def load():
            m=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='eager').eval();p=ColPaliProcessor.from_pretrained(adapter,local_files_only=True);return m,p
    else:load=env['load_visual']
    model,processor=measure('model_load',load,profiled=proc==3)
    if model_name=='ColPali':C.write(C.A/'trained_adapter_check.json',validate_model(model,overlay,mapping))
    bm=measure('BM25_build',lambda:env['BM25'](pages));frozen_index=C.ROOT/'cache/indexes'/('icde_eab_independent_007_biomedical_colqwen_001' if model_name=='ColQwen2' else 'icde_eab_lastmile_008_biomedical_colpali_001')
    newbase=C.ROOT/'cache/indexes'/f'{C.RUN}_{model_name}_unprofiled_001';index=newbase if proc<3 else C.ROOT/'cache/indexes'/f'{C.RUN}_{model_name}_profiled_001'
    equivalence=[];block_stats=[]
    if proc in [0,3]:
        index.mkdir(exist_ok=False);(index/'visual_parts').mkdir();old_manifest=C.load(frozen_index/'index_manifest.json','IMMUTABLE_OLD_INDEX_METADATA');old_values={}
        for part in sorted((frozen_index/'visual_parts').glob('*.pt')):
            assert C.sha(part)==old_manifest['files'][str(part.relative_to(frozen_index))];d=torch.load(part,weights_only=True);old_values.update(zip(d['ids'],d['embeddings']))
        done=[];corpus=C.ROOT/bio['official_inputs']['corpus']['path']
        def build():
            for bi,batch in enumerate(pq.ParquetFile(corpus).iter_batches(batch_size=32,columns=['corpus-id','image'])):
                rr=batch.to_pylist()
                def block():
                    for j in range(0,len(rr),2):
                        guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30
                        x=rr[j:j+2];images=[Image.open(io.BytesIO(z['image']['bytes'])).convert('RGB') for z in x];inp=processor.process_images(images).to('cuda:0');em=model(**inp);valid=inp['attention_mask'].bool();vectors=[v[m].detach().to(torch.float16).cpu() for v,m in zip(em,valid)];pids=[int(z['corpus-id']) for z in x]
                        for pid,v in zip(pids,vectors):
                            ov=old_values[pid];assert v.shape==ov.shape;error=float((v.float()-ov.float()).abs().max());assert error<=1e-3,'NEW_INDEX_VECTOR_MISMATCH_NO_QUALITY_REPLACEMENT';equivalence.append({'corpus_id':pid,'maxabs':error,'bitwise_equal':torch.equal(v,ov)})
                        a=time.perf_counter();torch.save({'ids':pids,'embeddings':vectors},index/'visual_parts'/f'{pids[0]:06d}.pt');block_stats.append({'kind':'serialization','wall_s':time.perf_counter()-a});done.extend(pids)
                if proc==3:
                    _,g=profile(block,C.A/f'index_block_{bi:03d}_GPU.json');block_stats.append({'kind':'encoding_block_profiled','pages':len(rr),**g})
                else:block()
                if (bi+1)%4==0:print(json.dumps({'model':model_name,'process':proc,'new_index_pages':len(done),'total':1016,'time_bj':C.stamp()}),flush=True)
            assert set(done)==set(ids)
        with torch.inference_mode():measure('full_index_build_and_persist',build)
        C.write(index/'index_manifest.json',{'pages':1016,'model':model_name,'cost_cohort':'UNPROFILED' if proc==0 else 'PROFILED','files':{str(p.relative_to(index)):C.sha(p) for p in index.rglob('*.pt')},'reference_original_manifest_SHA256':C.sha(frozen_index/'index_manifest.json'),'weights_and_processor':'unchanged;001/003 pinned loaders','scientific_quality_outputs_replaced':False})
        C.table(C.R/'index_vector_equivalence.csv',equivalence);C.table(C.R/'index_block_costs.csv',block_stats);del old_values
    im=C.load(index/'index_manifest.json','NEW_COST_INDEX_METADATA');vv={}
    def resident():
        for part in sorted((index/'visual_parts').glob('*.pt')):
            guard.check();assert C.sha(part)==im['files'][str(part.relative_to(index))];z=torch.load(part,weights_only=True);vv.update(zip(z['ids'],z['embeddings']))
        lengths=torch.tensor([len(vv[i]) for i in ids],device='cuda:0');passage=torch.nn.utils.rnn.pad_sequence([vv[i].float() for i in ids],batch_first=True).cuda();return lengths,passage
    lengths,passage=measure('index_residency',resident,profiled=proc==3)
    eng=type('StaticEngine',(),{})();eng.torch=torch;eng.passage=passage;eng.lengths=lengths;eng.ids=ids;eng.env=env
    def encode(text):return model(**processor.process_queries([text]).to('cuda:0'))[0].float()
    def pali_score(q,indices):
        values=[]
        for st in range(0,len(indices),64):
            ix=indices[st:st+64];t=torch.tensor(ix,device='cuda:0');block=passage.index_select(0,t);logits=torch.matmul(q.unsqueeze(0),block.transpose(1,2));mask=torch.arange(passage.shape[1],device='cuda:0')[None,:]>=lengths.index_select(0,t)[:,None];logits.masked_fill_(mask[:,None,:],-torch.inf);values+=logits.max(2).values.sum(1).cpu().tolist()
        return [indices[i] for i in stable_ranking([ids[i] for i in indices],values)]
    score=(lambda q,ix:E.Engine.score(eng,q,ix)) if model_name=='ColQwen2' else pali_score
    def execute(plan,text):
        if plan=='Full-Visual':candidate=list(range(len(ids)))
        else:
            order,_=bm.search(text)
            if plan=='BM25':return [ids[i] for i in order[:10]]
            candidate=order[:int(plan[6:])]
        if not candidate:return []
        return [ids[i] for i in score(encode(text),candidate)[:10]]
    requests=[];first=[]
    with torch.inference_mode():
        for plan in C.PLANS:
            st=time.perf_counter();ct=time.process_time();execute(plan,queries[bio['query_ids'][0]]);torch.cuda.synchronize();first.append({'plan':plan,'first_request_wall_s':time.perf_counter()-st,'first_request_CPU_s':time.process_time()-ct,'process_cold_to_first_s':time.perf_counter()-EARLY,'operational_cold_to_first_s':time.perf_counter()-operational_start,'includes_index_build':proc in [0,3]})
        for plan in C.PLANS:
            for _ in range(3):execute(plan,queries[bio['query_ids'][0]])
        if proc<3:
            work=[(q,p) for q in bio['query_ids'] for p in C.PLANS];np.random.default_rng(C.SEED+proc).shuffle(work)
            for q,plan in work:
                guard.check();torch.cuda.synchronize();st=time.perf_counter();ct=time.process_time();out=execute(plan,queries[q]);torch.cuda.synchronize();requests.append({'model':model_name,'process':proc,'query_id':q,'plan':plan,'E2E_s':time.perf_counter()-st,'CPU_s':time.process_time()-ct,'cohort':'NEW009_UNPROFILED_COST_ONLY','output_top10':out,'quality_labels_read':False})
        else:
            prof=[]
            for q in bio['query_ids'][:12]:
                for plan in C.PLANS:
                    if plan=='BM25':continue
                    _,g=profile(lambda:execute(plan,queries[q]),C.A/f'warm_q{q}_{plan}_GPU.json',require_kernels=True);prof.append({'query_id':q,'plan':plan,'scope':'first12 fixed IDs technical cost profile, not160 mean',**g})
            C.table(C.R/'warm_GPU_profiles.csv',prof)
    C.table(C.R/'setup_components.csv',records);C.table(C.R/'first_requests.csv',first)
    if requests:C.table(C.R/'requests.csv',requests)
    C.write(C.A/'completed.json',{'status':'PASS_COST_ONLY','model':model_name,'process':proc,'time_bj':C.stamp(),'request_count':len(requests),'queries':160 if requests else 12,'index':str(index.relative_to(C.ROOT)),'index_bytes':sum(p.stat().st_size for p in index.rglob('*') if p.is_file()),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'process_total_wall_s':time.perf_counter()-EARLY,'qrels_reads':0,'sealed_reads':0,'OS_cache':'not flushed; processcold only','energy':'NOT_MEASURED'})
    print(json.dumps({'status':'PASS_COST_WORKER','model':model_name,'process':proc,'requests':len(requests),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
