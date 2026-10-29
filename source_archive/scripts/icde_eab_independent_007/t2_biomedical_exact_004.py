"""Frozen single-pass Biomedical evaluation. No fitting, no threshold selection."""
import common as C
import sys,os,time,json,io,shutil,resource,collections,hashlib
from pathlib import Path
import numpy as np
gpu=int(sys.argv[1]);proc=int(sys.argv[2]);C.start(f'T2/process_{proc}_001')
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import engine_exact_004 as E
E.idle(gpu)
sys.path.insert(0,str(C.ROOT/'src'))
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
from mdrouter_v5.graded_metrics import ndcg,b1l
guard=ResourceGuard(gpu,C.A/'resource_guard.json',period=.1)
try:
    import torch,pyarrow.parquet as pq
    from PIL import Image
    import learning_base_001 as L
    protocol=C.load(C.ART/'T1/biomedical_frozen_protocol.json','PREREGISTERED_PROTOCOL');assert protocol['status']=='FROZEN_BEFORE_RETRIEVAL_OUTCOMES'
    for p,h in protocol['code_SHA256'].items():assert C.sha(C.ROOT/p)==h,'preregistered code changed'
    index=C.ROOT/'cache/indexes/icde_eab_independent_007_biomedical_colqwen_001'
    if proc==0:
        index.mkdir(exist_ok=False);(index/'visual_parts').mkdir();pages=C.load(C.ART/'T1_preparation/ocr_001/pages.json','UNLABELLED_OCR');C.write(index/'pages.json',pages)
        env=E.backend();t=time.perf_counter();ct=time.process_time();model,processor=env['load_visual']();torch.cuda.synchronize();loadcost={'wall_s':time.perf_counter()-t,'CPU_s':time.process_time()-ct};assert processor.image_processor.size['longest_edge']==768*28*28
        corpus=C.ROOT/protocol['official_inputs']['corpus']['path'];assert C.sha(corpus)==protocol['official_inputs']['corpus']['sha256'];done=[];stats=[];t=time.perf_counter();ct=time.process_time()
        with torch.inference_mode():
            for bi,batch in enumerate(pq.ParquetFile(corpus).iter_batches(batch_size=64,columns=['corpus-id','image'])):
                rows=batch.to_pylist()
                def block():
                    for i in range(0,len(rows),2):
                        guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30;rr=rows[i:i+2];images=[Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in rr];inp=processor.process_images(images).to('cuda:0');em=model(**inp);valid=inp['attention_mask'].bool();ee=[e[m].detach().to(dtype=torch.float16).cpu() for e,m in zip(em,valid)];assert all(torch.isfinite(e).all() for e in ee);ids=[int(r['corpus-id']) for r in rr];torch.save({'ids':ids,'embeddings':ee},index/'visual_parts'/f'{ids[0]:06d}.pt');done.extend(ids)
                _,st=profile_call(block,C.A/f'index_block_{bi:03d}.json');stats.append(st);print(json.dumps({'Bio_index_pages':len(done),'total':1016,'time_bj':C.stamp()}),flush=True)
        assert set(done)=={int(p['corpus_id']) for p in pages}
        C.write(index/'index_manifest.json',{'pages':1016,'model_load':loadcost,'setup_wall_s':time.perf_counter()-t,'setup_CPU_s':time.process_time()-ct,'setup_GPU_activity_s':sum(x['gpu_active_union_seconds'] for x in stats),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'files':{str(p.relative_to(index)):C.sha(p) for p in index.rglob('*') if p.is_file()},'OCR_setup':'see T1_preparation/ocr_001/setup_cost.json','model_revision':protocol['primary_model_revision']});C.table(C.R/'index_build_gpu_profiles.csv',stats)
        del model,processor;torch.cuda.empty_cache()
    eng=E.Engine(index.name,guard);torch.cuda.synchronize();C.write(C.A/'setup.json',eng.setup)
    # Verify implementation against the official processor, without selecting any operating point.
    test_query='What is the structure of a biological cell?'
    with torch.inference_mode():
        qe=eng.encode(test_query);v=eng.passage[:12];logits=torch.matmul(qe.unsqueeze(0),v.transpose(1,2));mask=torch.arange(v.shape[1],device='cuda:0')[None,:]>=eng.lengths[:12,None];ours=logits.masked_fill(mask[:,None,:],-torch.inf).max(2).values.sum(1).cpu().numpy();official=np.array([eng.processor.score_multi_vector([qe],[v[i,:eng.page_lengths[i]]],device='cuda:0')[0,0].item() for i in range(12)]);assert np.max(np.abs(ours-official))<1e-3
    raw=C.ROOT/'artifacts/icde_eab_main_006/X3/biomedical_audit_inputs_001'
    def read(kind):
        path=next((raw/kind).glob('*.parquet'));assert C.sha(path)==protocol['official_inputs'][kind]['sha256']
        with (C.A/'biomedical_access_ledger.jsonl').open('a') as f:f.write(json.dumps({'time_bj':C.stamp(),'caller':str(__file__),'pid':os.getpid(),'role':'AUTHORIZED_PREREGISTERED_PUBLIC_BENCHMARK','path':str(path),'sha256':C.sha(path),'not_project_sealed_test':True})+'\n')
        return pq.read_table(path).to_pylist()
    readwall=time.perf_counter();readcpu=time.process_time();queries={int(x['query-id']):x['query'] for x in read('queries')};labels=read('qrels');ql=collections.defaultdict(dict)
    for r in labels:ql[int(r['query-id'])][int(r['corpus-id'])]=int(r['score'])
    C.write(C.A/'label_materialization_cost.json',{'wall_s':time.perf_counter()-readwall,'CPU_s':time.process_time()-readcpu,'GPU_s':0,'scope':'loading existing official public query/qrels files; no human annotation cost measured'})
    ids=protocol['query_ids'];assert sorted(queries)==sorted(ids) and len(ids)==160
    groups={int(x['query_id']):x for x in C.load(C.ROOT/protocol['query_family_map'])}
    locked=C.load(C.ART/'T0/locked_models.json','FROZEN_FIT81_MODELS');cvec=np.array(protocol['inherited_cost_vector']);catalog=np.array([0,2,3])
    models={x['method']:x for x in locked}
    rr=C.load('artifacts/icde_evidence_closure_003/E2/candidate_models.json','FROZEN_FIT81_MODELS')
    for target in ['hard','soft']:models['RR_'+target]=next(x for x in rr if x['source']=='Pharma_only' and x['seed']==202610271130 and x['method']=='Q_'+target and x['kappa']==20)
    def unpack(m):
        a=[np.array(x) for x in m['model']];a[0]=a[0].astype(bool);return tuple(a)
    models={k:unpack(v) for k,v in models.items()};costnorm=cvec/float(cvec[0]) if protocol['cost_vector_normalize_by_FullV'] else cvec
    methods=C.PLANS+['Q_regret','QP_regret','RandomMix','RR_hard','RR_soft']
    outputs={};reference={};correct=[];quality=[];actions=[];rows=[];profiles=[]
    def route(method,qid,detail=False):
        text=queries[qid];probe=None;action=method;qw=pw=statsw=planner=0.
        if method in ['Q_regret','QP_regret','RR_hard','RR_soft']:
            t=time.perf_counter();q=eng.Q(text);qw=time.perf_counter()-t
            if method=='QP_regret':
                t=time.perf_counter();probe=eng.bm.search(text);pw=time.perf_counter()-t;t=time.perf_counter();p=E.probe_features(eng.bm,text,*probe);statsw=time.perf_counter()-t;x=np.r_[q,p]
            else:x=q
            t=time.perf_counter();pred=L.predict(models[method],x[None])[0];assert np.isfinite(pred).all()
            ai=int(catalog[np.argmin(costnorm[catalog]+20*np.maximum(pred[catalog],0))]) if method.endswith('regret') else int(catalog[np.argmax(pred[catalog])]);action=C.PLANS[ai];planner=time.perf_counter()-t
        elif method=='RandomMix':
            t=time.perf_counter();h=hashlib.sha256(f'{C.SEED}:Biomedical:{qid}:RandomMix'.encode()).digest();u=int.from_bytes(h[:8],'big')/2**64;ai=int(np.searchsorted(np.cumsum(protocol['random_mix_probabilities']),u,side='right'));action=C.PLANS[min(ai,3)];planner=time.perf_counter()-t
        t=time.perf_counter();result=eng.execute(action,text,probe=probe);pathwall=time.perf_counter()-t
        return result,{'action':action,'Q_wall_s':qw,'Probe_extra_wall_s':pw,'Probe_statistics_wall_s':statsw,'planner_wall_s':planner,'selected_path_wall_s':pathwall,'Probe_reused':probe is not None and action!='Full-Visual','Probe_paid_before_FullV':probe is not None and action=='Full-Visual'}
    with torch.inference_mode():
        for qid in ids:
            ref=eng.execute('Full-Visual',queries[qid]);reference[qid]=ref;bmorder,_=eng.bm.search(queries[qid]);u0=ndcg(ref,ql[qid]);positives={p for p,g in ql[qid].items() if g>0}
            for method in methods:
                result,d=route(method,qid);outputs[qid,method]=result;action=d['action'];cand=bmorder[:int(action[6:])] if action.startswith('BM25-R') else list(range(len(eng.ids)));contained=set(ref)<=set(eng.ids[j] for j in cand);assert action not in ['BM25-R50','BM25-R100'] or not contained or result==ref
                u=ndcg(result,ql[qid]);r=max(u0-u,0);rb=0 if action=='Full-Visual' else b1l(ref,result,ql[qid]);quality.append({'query_id':qid,'family':groups[qid]['query_family'],'source_group':groups[qid]['source_group'],'method':method,'action':action,'nDCG10':u,'reference_nDCG10':u0,'r':r,'signed_gap':u0-u,'r_bar':rb,'harm':int(r>1e-10),'severity_ge_01':int(r>=.1),'BM25_Recall50':len(positives&{eng.ids[j] for j in bmorder[:50]})/len(positives),'reference_top10_in_candidates':contained,'ranking':result,'reference_ranking':ref,'evidence':'PREREGISTERED_FROZEN_POLICY_DESCRIPTIVE','certification':'NOT_CERTIFIED'})
                actions.append({'query_id':qid,'method':method,**d});correct.append({'query_id':qid,'method':method,'same_scorer_containment_check':True,'status':'PASS'})
            if len(reference)%20==0:print(json.dumps({'Bio_quality_queries':len(reference),'total':160,'time_bj':C.stamp()}),flush=True)
        C.table(C.R/'quality.csv',quality);C.table(C.R/'actions.csv',actions);C.table(C.R/'correctness.csv',correct)
        for method in methods:
            for _ in range(3):route(method,ids[0])
        work=[(q,m,k) for q in ids for m in methods for k in range(3)];np.random.default_rng(C.SEED+proc).shuffle(work)
        for wi,(qid,method,k) in enumerate(work):
            guard.check();torch.cuda.synchronize();t=time.perf_counter();ct=time.process_time();tt=time.thread_time();result,d=route(method,qid);torch.cuda.synchronize();wall=time.perf_counter()-t;cpu=time.process_time()-ct;thread=time.thread_time()-tt;assert result==outputs[qid,method];rows.append({'process':proc,'query_id':qid,'method':method,'repeat':k,'E2E_s':wall,'CPU_process_s':cpu,'CPU_thread_s':thread,**d,'cohort':'UNPROFILED_ACTUAL_COMPLETE_PATH'})
            if (wi+1)%500==0:print(json.dumps({'Bio_timed_requests':wi+1,'total':len(work),'process':proc,'time_bj':C.stamp()}),flush=True)
        C.table(C.R/'requests.csv',rows)
        if proc==0:
            for qi,qid in enumerate(ids):
                for method in methods:
                    guard.check();act=next(x['action'] for x in actions if x['query_id']==qid and x['method']==method);needs_gpu=act=='Full-Visual' or (act!='BM25' and bool(eng.bm.search(queries[qid])[0]));_,stat=profile_call(lambda:route(method,qid),C.A/f'profile_whole_q{qid}_{method}.json',require_kernels=needs_gpu);profiles.append({'query_id':qid,'method':method,'component':'complete_route',**stat})
                _,enc=profile_call(lambda:eng.encode(queries[qid]),C.A/f'profile_encoding_q{qid}.json');profiles.append({'query_id':qid,'method':'shared_visual','component':'query_encoding',**enc});qe=eng.encode(queries[qid]);order,_=eng.bm.search(queries[qid])
                for plan in ['Full-Visual','BM25-R50','BM25-R100']:
                    cand=list(range(len(eng.ids))) if plan=='Full-Visual' else order[:int(plan[6:])];_,ret=profile_call(lambda:eng.score(qe,cand),C.A/f'profile_scoring_q{qid}_{plan}.json',require_kernels=bool(cand));profiles.append({'query_id':qid,'method':plan,'component':'retrieval_scoring',**ret})
                if (qi+1)%20==0:print(json.dumps({'Bio_profiled_queries':qi+1,'total':160,'time_bj':C.stamp()}),flush=True)
            C.table(C.R/'GPU_profiles.csv',profiles)
    C.write(C.A/'completed.json',{'status':'PASS_ACTUAL_NEW_BENCHMARK_EXECUTION','unique_queries':160,'timed_requests':len(rows),'process':proc,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'time_bj':C.stamp()})
finally:guard.close()
