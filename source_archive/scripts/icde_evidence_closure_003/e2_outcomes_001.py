import common as C
import engine_001 as E
import numpy as np,sys,time,random,json,collections,resource
gpu=sys.argv[1];C.start('E2/outcomes_001');C.STAGE='E2_OUTCOMES';E.idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
from mdrouter_v5.graded_metrics import ndcg,b1l
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import torch
 queries=C.load(C.ART/'E2/allowed_fitdev_queries.json','FROZEN_FITDEV_ALLOWLIST');allowed={int(q['query_id']) for q in queries};projection=C.load(C.ART/'E2/public_projection_final_001/projection_manifest.json','PUBLIC_PROJECTION_SCOPE_AUDIT');assert set(projection['allowed_ids'])==allowed and projection['Cal_Eval_records_delivered']==0
 qrels=C.load(C.ART/'E2/public_projection_final_001/fitdev_qrels.json','AUTHORIZED_FITDEV_QRELS_ONLY');assert all(int(r['query_id']) in allowed for r in qrels);assert C.sha(C.ART/'E2/public_projection_final_001/fitdev_qrels.json')==projection['fitdev_qrels_SHA256'];ql=collections.defaultdict(dict)
 for r in qrels:ql[int(r['query_id'])][int(r['corpus_id'])]=int(r['score'])
 assert set(ql)==allowed;eng=E.Engine('pharma_fixedcorpus_index_001',guard);plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];rows=[];profiles=[];features=[];ranks={};raw=[]
 with torch.inference_mode():
  for p in plans:eng.execute(p,queries[0]['query'])
  for repeat in range(3):
   work=queries.copy();random.Random(C.SEED+repeat).shuffle(work)
   for qi,q in enumerate(work):
    for p in random.Random(C.SEED+repeat*131+qi).sample(plans,len(plans)):
     guard.check();torch.cuda.synchronize();ct=time.process_time();tw=time.perf_counter();top=eng.execute(p,q['query']);torch.cuda.synchronize();row={'query_id':q['query_id'],'role':q['role'],'plan':p,'repeat':repeat,'CPU_s':time.process_time()-ct,'resident_E2E_s':time.perf_counter()-tw,'top10':top};raw.append(row)
     with (C.A/'requests_partial.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     key=(int(q['query_id']),p)
     if key in ranks:assert ranks[key]==top
     else:ranks[key]=top
   print(json.dumps({'stage':'E2_OUTCOMES','repeat':repeat,'requests':len(raw),'time_bj':C.stamp()}),flush=True)
  for i,q in enumerate(queries):
   qid=int(q['query_id']);Qcost=[];Pcost=[];Sinfo=[];Pinfo=[]
   for _ in range(3):
    ct=time.process_time();tw=time.perf_counter();Q=eng.Q(q['query']);Qcost.append((time.process_time()-ct,time.perf_counter()-tw));ct=time.process_time();tw=time.perf_counter();probe=eng.bm.search(q['query']);Sinfo.append((time.process_time()-ct,time.perf_counter()-tw));ct=time.process_time();tw=time.perf_counter();P=E.probe_features(eng.bm,q['query'],*probe);Pinfo.append((time.process_time()-ct,time.perf_counter()-tw));Pcost.append(tuple(np.array(Sinfo[-1])+Pinfo[-1]))
   features.append({'query_id':qid,'role':q['role'],'family':q['family_id'],'generator':q['generator'],'Q':Q.tolist(),'P':P.tolist(),'Q_CPU_s':np.mean(Qcost,axis=0)[0],'Q_wall_s':np.mean(Qcost,axis=0)[1],'Probe_CPU_s':np.mean(Pcost,axis=0)[0],'Probe_wall_s':np.mean(Pcost,axis=0)[1],'BM_search_wall_s':np.mean(Sinfo,axis=0)[1],'Pstats_wall_s':np.mean(Pinfo,axis=0)[1]})
   for p in plans:
    guard.check();_,st=profile_call(lambda:eng.execute(p,q['query']),C.A/f'trace_{qid}_{p}.json',require_kernels=p!='BM25');profiles.append({'query_id':qid,'plan':p,**st});top=ranks[qid,p];ref=ranks[qid,'Full-Visual'];Uref=ndcg(ref,ql[qid]);U=ndcg(top,ql[qid]);rb=b1l(ref,top,{pid:ql[qid].get(pid,0) for pid in ref});r=max(Uref-U,0);assert r<=rb+1e-8
    rr=[x for x in raw if x['query_id']==qid and x['plan']==p];cand=probe[0][:int(p[6:])] if p.startswith('BM25-R') else probe[0] if p=='BM25' else list(range(len(eng.ids)));cids={eng.ids[x] for x in cand}
    if p.startswith('BM25-R') and set(ref)<=cids:assert top==ref,'Full reference Top10 coverage must imply exact rerank agreement'
    rows.append({'domain':'Pharmaceuticals','query_id':qid,'role':q['role'],'family':q['family_id'],'plan':p,'nDCG10':U,'U_reference':Uref,'r':r,'r_bar':rb,'signed_gap':Uref-U,'harm':r>1e-10,'gold_relevant_n':len(ql[qid]),'gold_recall_at_candidate_k':len(set(ql[qid])&cids)/len(ql[qid]),'reference_top10_full':set(ref)<=cids,'reference_positive_top10_missing':len({x for x in ref if ql[qid].get(x,0)>0}-cids),'GPU_activity_s':st['gpu_active_union_seconds'],'mean_CPU_s':np.mean([x['CPU_s'] for x in rr]),'mean_E2E_s':np.mean([x['resident_E2E_s'] for x in rr]),'status':'AUTHORIZED_FITDEV_EXPLORATORY_NOT_CERTIFIED'})
   if (i+1)%10==0:print(json.dumps({'stage':'E2_OUTCOMES','profiled_queries':i+1,'time_bj':C.stamp()}),flush=True)
 C.write(C.A/'features.json',features);C.write(C.A/'outcomes.json',rows);C.write(C.A/'request_rows.json',raw);C.write(C.A/'profile_rows.json',profiles);C.write(C.A/'setup.json',eng.setup);C.table(C.R/'per_query_plan.csv',rows);print(json.dumps({'status':'PASS_FIT81_DEV45_OUTCOMES','rows':len(rows),'requests':len(raw),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
