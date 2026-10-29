import common as C
import engine_001 as E
import learning_base_001 as L
from idle_wait_004 import idle
import numpy as np,sys,os,time,json,random,concurrent.futures,resource
d,gpu,rep_s=sys.argv[1:];rep=int(rep_s);C.start(f'S2/resident_workers/{d}_p{rep}_001');os.sched_setaffinity(0,{'1':[4,5,24,25],'4':[10,11,30,31]}[gpu]);idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import torch
 models=C.load(C.ART/'S0/locked_models.json');md={m['method']:m for m in models};fm={}
 for k,m in md.items():v=[np.array(x) for x in m['model']];v[0]=v[0].astype(bool);fm[k]=v
 dc=next(x for x in C.load(C.ART/'S0/domain_policy_config.json') if x['domain']==d);cost=np.array(dc['label_free_cost_vector']);methods=C.PLANS+['Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret']
 if d=='Pharmaceuticals':
  qs=[q for q in C.load(C.PREV/'E2/allowed_fitdev_queries.json') if q['role']=='development'];rr=C.load(C.PREV/'E2/outcomes_001/request_rows.json');expected={(int(r['query_id']),r['plan']):r['top10'] for r in rr if r['role']=='development'};qr=C.load(C.PREV/'E2/outcomes_001/outcomes.json');lookup={(int(r['query_id']),r['plan']):r for r in qr if r['role']=='development'};groups={int(q['query_id']):q['family_id'] for q in qs};index='pharma_fixedcorpus_index_001'
 else:
  qs=C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json');expected={(int(r['query_id']),r['plan']):r['top10'] for r in C.load(f'artifacts/icde_decisive_suite_002/R2/{d}_candidate_rankings.json')};qr=C.load(C.PR/'E1/per_query_quality_cost.csv');lookup={(int(r['query_id']),r['plan']):r for r in qr if r['domain']==d};groups={int(r['query_id']):r['family'] for r in qr if r['domain']==d};index=dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']))[d]
 tw=time.perf_counter();eng=E.Engine(index,guard);torch.cuda.synchronize();cold=time.perf_counter()-tw;rows=[];profiles=[];pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);rng=np.random.default_rng(C.SEED+rep);partial=[]
 def invoke(q,method,repeat,profiled=False):
  guard.check();torch.cuda.synchronize();ct=time.thread_time();start=time.perf_counter();pb=None;bmwall=pwall=qwall=decisionwall=0.
  if method in C.PLANS:act=C.PLANS.index(method)
  elif method.startswith('RandomMix_'):act=int(rng.choice(4,p=dc['mixes'][method]))
  else:
   t=time.perf_counter();x=eng.Q(q['query']);qwall=time.perf_counter()-t
   if method=='QP_regret':
    t=time.perf_counter();pb=eng.bm.search(q['query']);bmwall=time.perf_counter()-t;t=time.perf_counter();p=E.probe_features(eng.bm,q['query'],*pb);pwall=time.perf_counter()-t;x=np.r_[x,p]
   t=time.perf_counter();pred=L.predict(fm[method],x[None])[0];v=cost+md[method]['kappa']*np.maximum(pred,0);v[1]=np.inf;act=int(v.argmin());decisionwall=time.perf_counter()-t
  pt=time.perf_counter()
  with torch.inference_mode():top=eng.execute(C.PLANS[act],q['query'],pb);torch.cuda.synchronize()
  stop=time.perf_counter();cpu=time.thread_time()-ct;qid=int(q['query_id']);assert top==expected[qid,C.PLANS[act]];z=lookup[qid,C.PLANS[act]]
  r={'domain':d,'process':rep,'query_id':qid,'family':groups[qid],'method':method,'repeat':repeat,'action':C.PLANS[act],'r':float(z['r']),'nDCG10':float(z['nDCG10']),'resident_full_path_E2E_s':stop-start,'CPU_thread_s':cpu,'Q_wall_s':qwall,'Probe_search_extra_wall_s':bmwall,'Pstats_extra_wall_s':pwall,'prediction_decision_wall_s':decisionwall,'selected_path_wall_s':stop-pt,'Probe_reused':pb is not None and act!=0,'Probe_paid_before_FullV':pb is not None and act==0,'native_thread':__import__('threading').get_native_id(),'cohort':'PROFILED_SECONDARY' if profiled else 'UNPROFILED_PRIMARY','certification':'NOT_CERTIFIED'};partial.append(r);return r
 for i in range(20):pool.submit(invoke,qs[i%len(qs)],'Full-Visual',-1).result()
 for m in methods:pool.submit(invoke,qs[0],m,-1).result()
 for repeat in range(3):
  work=qs.copy();random.Random(C.SEED+rep*100+repeat).shuffle(work)
  for qi,q in enumerate(work):
   order=methods.copy();random.Random(C.SEED+rep*1000+repeat*100+qi).shuffle(order)
   for m in order:rows.append(pool.submit(invoke,q,m,repeat).result())
   with (C.A/'requests_partial.jsonl').open('a') as f:
    for r in rows[-len(methods):]:f.write(json.dumps(C.clean(r))+'\n')
  print(json.dumps({'domain':d,'process':rep,'repeat':repeat,'requests':len(rows),'time_bj':C.stamp()}),flush=True)
 C.write(C.A/'request_rows.json',rows);C.write(C.A/'setup.json',{'cold_engine_wall_s':cold,'engine_setup':eng.setup,'CPU_affinity':sorted(os.sched_getaffinity(0)),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'host_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 # Independent GPU profile cohort; no profiled latency enters primary figures.
 if rep==0:
  for q in qs:
   for m in methods:
    def call():return pool.submit(invoke,q,m,99,True).result()
    result,st=profile_call(call,C.A/f'GPU_{q["query_id"]}_{m}.json',require_kernels=m=='Full-Visual');profiles.append({'domain':d,'query_id':q['query_id'],'method':m,'action':result['action'],**st})
 C.write(C.A/'GPU_profiles.json',profiles);pool.shutdown(wait=True)
finally:
 C.write(C.A/'completed_invokes.json',partial if 'partial' in globals() else []);guard.close()
print(json.dumps({'status':'PASS_RESIDENT_FULL_PATH','domain':d,'process':rep,'requests':len(rows),'GPU_profiles':len(profiles),'time_bj':C.stamp()}),flush=True)
