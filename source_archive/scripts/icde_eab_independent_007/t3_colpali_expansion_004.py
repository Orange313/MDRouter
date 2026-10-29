"""Pinned ColPali pilot. Only predeclared Dev queries, unlabeled corpus pages."""
import common as C
import os, sys, time, json, io, shutil, subprocess, resource, ast, collections, math
from pathlib import Path
import numpy as np
gpu=int(sys.argv[1]);domain=sys.argv[2];proc=int(sys.argv[3])
stage=f'T3/expansion_{domain}_p{proc}_004';C.start(stage)
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
    primary={'vidore_v3_hr':'hr_primary_001','vidore_v3_finance_en':'finance_primary_001','vidore_v3_industrial':'industrial_primary_003','Pharmaceuticals':'pharma_fixedcorpus_index_001'}[domain]
    pages=C.load(C.ROOT/'cache/indexes'/primary/'pages.json','UNLABELLED_CORPUS_TEXT');pages.sort(key=lambda r:int(r['corpus_id']));ids=[int(r['corpus_id']) for r in pages];pos={v:i for i,v in enumerate(ids)}
    nodes=[x for x in ast.parse(C.load('scripts/cross_domain_four_plan_001.py')).body if isinstance(x,ast.ClassDef) and x.name=='BM25'];env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'pinned_BM25','exec'),env);t=time.perf_counter();bm=env['BM25'](pages);setup['BM25_setup_wall_s']=time.perf_counter()-t
    inventory=C.load('manifests/primary_datasets_inventory.json')
    if domain=='Pharmaceuticals':
        ds={'revision':'262e203d7c59c55947f7042812e0bcc1eee190b1','files':[{'path':'corpus/'+p.name,'lfs_sha256':h} for p,h in zip(sorted((C.ROOT/'data/primary/vidore_v3_pharmaceuticals/262e203d7c59c55947f7042812e0bcc1eee190b1/corpus').glob('*.parquet')),['0f51d221a1def60c6d1436a31507e69be42a6a1da4a9ae45ff6e231751466ac5','827305c58164c9e69581953560b1e76349dbc86a5a06df91d16c92023175e468'])]};corpus=C.ROOT/'data/primary/vidore_v3_pharmaceuticals'/ds['revision']/'corpus'
    else:
        ds=next(x for x in inventory['datasets'] if x['repo'].split('/')[-1]==domain);corpus=C.ROOT/'data/primary'/domain/ds['revision']/'corpus'
    index=C.ROOT/'cache/indexes'/f'{C.RUN}_colpali_{domain}_002'
    if proc==0 and not index.exists():
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
    lengths=torch.tensor([len(vv[i]) for i in ids],device='cuda:0');passage=torch.nn.utils.rnn.pad_sequence([vv[i].float() for i in ids],batch_first=True).cuda();torch.cuda.synchronize();setup['index_residency_wall_s']=time.perf_counter()-t;setup['index_manifest_SHA256']=C.sha(index/'index_manifest.json');C.write(C.A/'setup.json',setup)
    if domain=='Pharmaceuticals':
        queries=[q for q in C.load('artifacts/icde_evidence_closure_003/E2/allowed_fitdev_queries.json') if q['role'].lower()=='dev'];labels=C.load('artifacts/icde_evidence_closure_003/E2/public_projection_final_001/fitdev_qrels.json','AUTHORIZED_FIT_DEV_ROLE_FILTERED')
    else:
        queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json');labels=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_qrels.json','AUTHORIZED_DEV_QRELS')
    queries={int(q['query_id']):q['query'] for q in queries};sel=sorted(queries);ql=collections.defaultdict(dict)
    for r in labels:
        if int(r['query_id']) in queries:ql[int(r['query_id'])][int(r['corpus_id'])]=int(r['score'])
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
            guard.check();torch.cuda.synchronize();st=time.perf_counter();ct=time.process_time();result=execute(plan,queries[qid]);torch.cuda.synchronize();cpu=time.process_time()-ct;wall=time.perf_counter()-st;assert result==outputs[qid,plan];u=ndcg(result,ql[qid]);u0=ndcg(outputs[qid,'Full-Visual'],ql[qid]);rows.append({'domain':domain,'process':proc,'query_id':qid,'plan':plan,'repeat':k,'nDCG10':u,'r':max(u0-u,0),'signed_gap':u0-u,'r_bar':0 if plan=='Full-Visual' else b1l(outputs[qid,'Full-Visual'],result,ql[qid]),'E2E_s':wall,'CPU_s':cpu,'probe_paid':'plan executes own BM25 search' if plan!='Full-Visual' else 'no BM probe','cohort':'UNPROFILED_PRIMARY','evidence':'DEV_EXPLORATORY'})
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
    C.write(C.A/'completed.json',{'status':'PASS_FOUR_DOMAIN_STATIC_EXPANSION','requests':len(rows),'queries':len(sel),'proc':proc,'time_bj':C.stamp(),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
finally:guard.close()
