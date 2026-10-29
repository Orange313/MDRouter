"""Pinned ColPali pilot. Only predeclared Dev queries, unlabeled corpus pages."""
import common as C
import os, sys, time, json, io, shutil, subprocess, resource, ast, collections, math
from pathlib import Path
import numpy as np
gpu=int(sys.argv[1]);proc=int(sys.argv[2]);domain='Biomedical'
C.start(f'L1/process_{proc}_001')
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
    t=time.perf_counter();loadcpu=time.process_time();model=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='eager').eval();processor=ColPaliProcessor.from_pretrained(str(adapter),local_files_only=True);torch.cuda.synchronize();setup={'model_load_wall_s':time.perf_counter()-t,'model_load_CPU_s':time.process_time()-loadcpu,'model_load_GPU_active_s':None,'model_load_GPU_scope':'not instrumented; missing not zero','active_adapters':model.active_adapters(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'processor_image_size':{'height':processor.image_processor.size.height,'width':processor.image_processor.size.width}}
    assert model.active_adapters(),'trained adapter not active';C.write(C.A/'adapter_loaded_tensors.json',validate_model(model,overlay,mapping))
    protocol=C.load(C.ART/'L1_preparation_001/protocol.json','POST_HOC_DESIGN_FIXED_BEFORE_COLPALI_OUTCOMES')
    bio=C.load(C.ROOT/'artifacts/icde_eab_independent_007/T1/biomedical_frozen_protocol.json','PRIOR_BENCHMARK_INPUT_LOCK')
    pages=C.load(C.ROOT/'artifacts/icde_eab_independent_007/T1_preparation/ocr_001/pages.json','SAME_UNLABELLED_OCR_1016_PAGES');pages.sort(key=lambda r:int(r['corpus_id']));ids=[int(r['corpus_id']) for r in pages];pos={v:i for i,v in enumerate(ids)}
    assert len(ids)==1016
    nodes=[x for x in ast.parse(C.load('scripts/cross_domain_four_plan_001.py')).body if isinstance(x,ast.ClassDef) and x.name=='BM25'];env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'pinned_BM25','exec'),env);t=time.perf_counter();bm=env['BM25'](pages);setup['BM25_setup_wall_s']=time.perf_counter()-t
    corpus_file=C.ROOT/bio['official_inputs']['corpus']['path'];assert C.sha(corpus_file)==bio['official_inputs']['corpus']['sha256']
    index=C.ROOT/'cache/indexes'/f'{C.RUN}_biomedical_colpali_001'
    if proc==0 and not index.exists():
        index.mkdir(exist_ok=False);(index/'visual_parts').mkdir();t=time.perf_counter();ct=time.process_time();done=[]
        setup_profiles=[]
        with torch.inference_mode():
            for bi,batch in enumerate(pq.ParquetFile(corpus_file).iter_batches(batch_size=32,columns=['corpus-id','image'])):
                rows=batch.to_pylist()
                def build_block():
                    for offset in range(0,len(rows),2):
                        guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30
                        rr=rows[offset:offset+2];imgs=[Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in rr];inp=processor.process_images(imgs).to('cuda:0');em=model(**inp);valid=inp['attention_mask'].bool();ee=[x[m].detach().to(dtype=torch.float16).cpu() for x,m in zip(em,valid)];assert all(torch.isfinite(x).all() for x in ee);rid=[int(r['corpus-id']) for r in rr];torch.save({'ids':rid,'embeddings':ee},index/'visual_parts'/f'{rid[0]:06d}.pt');done.extend(rid)
                _,stats=profile_call(build_block,C.A/f'index_gpu_block_{bi:03d}.json');setup_profiles.append(stats)
                print(json.dumps({'stage':f'L1/process_{proc}_001','pages':len(done),'total':len(ids),'wall_s':time.perf_counter()-t,'time_bj':C.stamp()}),flush=True)
        assert set(done)==set(ids)
        C.table(C.R/'index_GPU_profiles.csv',setup_profiles)
        C.write(index/'index_manifest.json',{'corpus_pages':len(ids),'setup_wall_s':time.perf_counter()-t,'setup_CPU_s':time.process_time()-ct,'setup_GPU_activity_s':sum(x['gpu_active_union_seconds'] for x in setup_profiles),'setup_GPU_scope':'MEASURED_PROFILED_PAGE_ENCODING_SETUP; includes profile overhead in setup wall','peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_host_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'files':{str(p.relative_to(index)):C.sha(p) for p in index.rglob('*.pt')},'model_preparation_SHA256':C.sha(C.ROOT/'artifacts/icde_eab_main_006/X1/preparation.json'),'OCR_shared_from007':'artifacts/icde_eab_independent_007/T1_preparation/ocr_001/setup_cost.json','processor':setup['processor_image_size'],'evidence':'EXPLORATORY_POSTHOC_EXTENDED'})
    im=C.load(index/'index_manifest.json','UNLABELLED_VISUAL_INDEX');vv={};t=time.perf_counter()
    for part in sorted((index/'visual_parts').glob('*.pt')):
        guard.check();assert C.sha(part)==im['files'][str(part.relative_to(index))];z=torch.load(part,weights_only=True);vv.update(zip(z['ids'],z['embeddings']))
    lengths=torch.tensor([len(vv[i]) for i in ids],device='cuda:0');passage=torch.nn.utils.rnn.pad_sequence([vv[i].float() for i in ids],batch_first=True).cuda();torch.cuda.synchronize();setup['index_residency_wall_s']=time.perf_counter()-t;setup['index_manifest_SHA256']=C.sha(index/'index_manifest.json');C.write(C.A/'setup.json',setup)
    def read_official(kind):
        path=C.ROOT/bio['official_inputs'][kind]['path'];assert C.sha(path)==bio['official_inputs'][kind]['sha256']
        with (C.A/'biomedical_access_ledger.jsonl').open('a') as f:f.write(json.dumps({'time_bj':C.stamp(),'caller':str(__file__),'path':str(path),'sha256':C.sha(path),'scope':'AUTHORIZED_PUBLIC_BENCHMARK_POST_HOC; NOT_SEALED_PROJECT_TEST'})+'\n')
        return pq.read_table(path).to_pylist()
    queries={int(q['query-id']):q['query'] for q in read_official('queries')};sel=sorted(queries);assert sel==bio['query_ids'] and len(sel)==160
    ql=collections.defaultdict(dict)
    for r in read_official('qrels'):ql[int(r['query-id'])][int(r['corpus-id'])]=int(r['score'])
    def score(q,indices):
        values=[]
        for s in range(0,len(indices),64):
            ix=indices[s:s+64];ix_t=torch.tensor(ix,device='cuda:0');block=passage.index_select(0,ix_t);logits=torch.matmul(q.unsqueeze(0),block.transpose(1,2));mask=torch.arange(passage.shape[1],device='cuda:0')[None,:]>=lengths.index_select(0,ix_t)[:,None];logits.masked_fill_(mask[:,None,:],-torch.inf);v=logits.max(2).values.sum(1);values+=v.cpu().tolist()
        order=stable_ranking([ids[i] for i in indices],values);return [indices[j] for j in order],values
    def execute(plan,text):
        if plan=='BM25':return [ids[i] for i in bm.search(text)[0][:10]]
        if plan=='Full-Visual':cand=list(range(len(ids)))
        else:order,_=bm.search(text);cand=order[:int(plan[6:])]
        if not cand:return []
        qe=model(**processor.process_queries([text]).to('cuda:0'))[0].float();ranking,_=score(qe,cand);return [ids[i] for i in ranking[:10]]
    correct=[];outputs={};rows=[];prof=[]
    plans=['Full-Visual','BM25-R50','BM25-R100','BM25']
    with torch.inference_mode():
        for qid in sel:
            text=queries[qid];qe=model(**processor.process_queries([text]).to('cuda:0'))[0].float();sample=list(range(min(12,len(ids))));ours,vals=score(qe,sample);official=processor.score_multi_vector([qe],[vv[ids[i]].float() for i in sample],device='cuda:0')[0].cpu().numpy();err=float(np.max(np.abs(np.array(vals)-official)));assert err<1e-3
            reference=execute('Full-Visual',text);order,_=bm.search(text)
            for plan in plans:
                result=execute(plan,text);cand=(order[:int(plan[6:])] if plan!='BM25' else order[:10]) if plan!='Full-Visual' else list(range(len(ids)));contained=set(reference)<=set(ids[i] for i in cand);assert plan=='BM25' or not contained or result==reference
                outputs[qid,plan]=result;correct.append({'domain':domain,'query_id':qid,'plan':plan,'processor_score_max_abs_error':err,'reference_top10_in_candidates':contained,'top10_equal_if_contained':None if not contained else result==reference,'output':result,'reference_output':reference,'BM25_positive_Recall50':len({ids[i] for i in order[:50]}&{p for p,g in ql[qid].items() if g>0})/len([g for g in ql[qid].values() if g>0]),'status':'PASS'})
        for plan in plans:
            for _ in range(3):execute(plan,queries[sel[0]])
        work=[(q,p,k) for q in sel for p in plans for k in range(3)];np.random.default_rng(C.SEED+proc).shuffle(work)
        for qid,plan,k in work:
            guard.check();torch.cuda.synchronize();st=time.perf_counter();ct=time.process_time();result=execute(plan,queries[qid]);torch.cuda.synchronize();cpu=time.process_time()-ct;wall=time.perf_counter()-st;assert result==outputs[qid,plan];u=ndcg(result,ql[qid]);u0=ndcg(outputs[qid,'Full-Visual'],ql[qid]);rows.append({'domain':domain,'process':proc,'query_id':qid,'plan':plan,'repeat':k,'nDCG10':u,'r':max(u0-u,0),'signed_gap':u0-u,'r_bar':0 if plan=='Full-Visual' else b1l(outputs[qid,'Full-Visual'],result,ql[qid]),'E2E_s':wall,'CPU_s':cpu,'probe_paid':'plan executes own BM25 search' if plan!='Full-Visual' else 'no BM probe','cohort':'UNPROFILED_PRIMARY','evidence':'EXPLORATORY_POSTHOC_EXTENDED'})
        if proc==0:
            for qid in sel:
                for plan in plans:
                    guard.check();_,stat=profile_call(lambda:execute(plan,queries[qid]),C.A/f'profile_q{qid}_{plan}.json',require_kernels=plan!='BM25');prof.append({'query_id':qid,'plan':plan,'component':'complete_path',**stat})
                    if plan=='BM25':continue
                    _,enc=profile_call(lambda:model(**processor.process_queries([queries[qid]]).to('cuda:0')),C.A/f'profile_encoding_q{qid}_{plan}.json');prof.append({'query_id':qid,'plan':plan,'component':'query_encoding',**enc})
                    query_embedding=model(**processor.process_queries([queries[qid]]).to('cuda:0'))[0].float();candidate=list(range(len(ids))) if plan=='Full-Visual' else bm.search(queries[qid])[0][:int(plan[6:])]
                    _,ret=profile_call(lambda:score(query_embedding,candidate),C.A/f'profile_retrieval_q{qid}_{plan}.json');prof.append({'query_id':qid,'plan':plan,'component':'retrieval_scoring',**ret})
    C.table(C.R/'requests.csv',rows);C.table(C.R/'correctness.csv',correct)
    if prof:C.table(C.R/'GPU_profiles.csv',prof)
    C.write(C.A/'completed.json',{'status':'PASS_BIOMEDICAL_COLPALI_STATIC_POSTHOC','requests':len(rows),'queries':len(sel),'proc':proc,'time_bj':C.stamp(),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
finally:guard.close()
