import common as C
import engine_001 as E
import learning_base_001 as L
import numpy as np,sys,time,threading,random,json,itertools,collections,concurrent.futures,ctypes as ct,os,resource
domain,gpu,mode=sys.argv[1:];tag=domain+'_'+mode+'_001';C.start('E4/workers/'+tag);C.STAGE='E4_'+tag;E.idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import torch
 protocol=C.load(C.ART/'E4/serving_protocol.json');dc=next(d for d in protocol['domains'] if d['domain']==domain);models=C.load(C.ART/'E4/locked_models.json');models={m['method']:m for m in models};fitmodels={}
 for name,m in models.items():
  z=[np.array(x) for x in m['model']];z[0]=z[0].astype(bool);fitmodels[name]=z
 plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];cost=np.array(dc['label_free_cost_vector'])
 if domain=='Pharmaceuticals':
  queries=[q for q in C.load(C.ART/'E2/allowed_fitdev_queries.json') if q['role']=='development'];expected={(int(r['query_id']),r['plan']):r['top10'] for r in C.load(C.ART/'E2/outcomes_001/request_rows.json') if r['role']=='development'};out=C.load(C.ART/'E2/outcomes_001/outcomes.json');risk={(int(r['query_id']),r['plan']):r['r'] for r in out if r['role']=='development'};groups={q['query_id']:q['family_id'] for q in queries};index='pharma_fixedcorpus_index_001'
 else:
  queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json');expected={(int(r['query_id']),r['plan']):r['top10'] for r in C.load(f'artifacts/icde_decisive_suite_002/R2/{domain}_candidate_rankings.json')};out=C.load('reports/icde_evidence_closure_003/E1/per_query_quality_cost.csv');risk={(int(r['query_id']),r['plan']):float(r['r']) for r in out if r['domain']==domain};a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{domain}_dataset.npz');groups=dict(zip(a['qids'],a['family_primary']));index=dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']))[domain]
 t=time.perf_counter();eng=E.Engine(index,guard);torch.cuda.synchronize();cold=time.perf_counter()-t;lock=threading.Lock();methodlist=['BM25','BM25-R50','BM25-R100','Full-Visual']+list(models)+list(dc['mixes']);N=64;requests=[];cells=[];profilerows=[]
 class Util(ct.Structure):_fields_=[('gpu',ct.c_uint),('memory',ct.c_uint)]
 def monitorutil():
  u=Util();code=guard.lib.nvmlDeviceGetUtilizationRates(guard.handle,ct.byref(u));return {'monotonic':time.perf_counter(),'SM_util_pct':u.gpu if code==0 else None,'memory_util_pct':u.memory if code==0 else None}
 def feature_decisions(batch,method,rng):
  ct0=time.thread_time();t0=time.perf_counter();probes=[None]*len(batch);probe_wall=0.
  if method in plans:act=np.repeat(plans.index(method),len(batch))
  elif method.startswith('RandomMix_'):act=rng.choice(4,len(batch),p=dc['mixes'][method])
  else:
   X=[]
   for i,q in enumerate(batch):
    x=eng.Q(q['query'])
    if method=='QP_regret':
     t=time.perf_counter();probes[i]=eng.bm.search(q['query']);p=E.probe_features(eng.bm,q['query'],*probes[i]);probe_wall+=time.perf_counter()-t;x=np.r_[x,p]
    X.append(x)
   pred=L.predict(fitmodels[method],np.array(X));utility=cost[None]+models[method]['kappa']*np.maximum(pred,0);utility[:,1]=np.inf;act=utility.argmin(1)
  return act,probes,{'planning_batch_CPU_s':time.thread_time()-ct0,'planning_batch_wall_s':time.perf_counter()-t0,'Probe_batch_wall_s':probe_wall}
 def run_cell(method,W,conc,load,profiled=False):
  # Identical sequence/arrivals across methods. Repeated workload requests are never new qrels samples.
  seed=C.SEED+W*100+conc*10+load;rr=random.Random(seed);ordered=queries.copy();rr.shuffle(ordered);work=[ordered[i%len(ordered)] for i in range(N)];rng=np.random.default_rng(seed+sum(map(ord,method)));qps=[.5,1.,2.][load]/dc['FullV_resident_mean_s'];start=time.perf_counter()+.02;allf=[];batch=[];batcharr=[];batchpositions=[];util=[];processcpu=time.process_time();cellrows=[];pcpu=0.;pwall=0.;probe=0.
  def execute(q,act,pb,arrival,dispatch,pos,overhead):
   ready=time.perf_counter();tc=time.thread_time();guard.check()
   with torch.inference_mode():
    if act==3:top=eng.execute(plans[act],q['query'],pb)
    else:
     with lock:top=eng.execute(plans[act],q['query'],pb);torch.cuda.synchronize()
   completed=time.perf_counter();cpu=time.thread_time()-tc;qid=int(q['query_id']);assert top==expected[qid,plans[act]],(domain,method,qid,'output mismatch')
   return {'domain':domain,'method':method,'window':W,'concurrency':conc,'load_index':load,'QPS_offered':qps,'position':pos,'query_id':qid,'family':groups[qid],'plan':plans[act],'r':risk[qid,plans[act]],'arrival_to_completion_s':completed-arrival,'window_and_planning_wait_s':dispatch-arrival,'executor_queue_s':ready-dispatch,'service_wall_s':completed-ready,'service_thread_CPU_s':cpu,'allocated_batch_CPU_s':overhead['planning_batch_CPU_s'],'allocated_batch_wall_s':overhead['planning_batch_wall_s'],'allocated_Probe_wall_s':overhead['Probe_batch_wall_s'],'Probe_reused':bool(pb is not None and act!=0),'Probe_paid_before_FullV':bool(pb is not None and act==0),'completed_time':completed,'cohort':'PROFILED_SECONDARY' if profiled else 'UNPROFILED_PRIMARY','certification':'NOT_CERTIFIED'}
  with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
   for i,q in enumerate(work):
    arrival=start+i/qps;wait=arrival-time.perf_counter()
    if wait>0:time.sleep(wait)
    batch.append(q);batcharr.append(arrival);batchpositions.append(i)
    if len(batch)==W or i==N-1:
     act,probes,ov=feature_decisions(batch,method,rng);pcpu+=ov['planning_batch_CPU_s'];pwall+=ov['planning_batch_wall_s'];probe+=ov['Probe_batch_wall_s'];dispatch=time.perf_counter();share={k:v/len(batch) for k,v in ov.items()}
     for q,a,pb,ar,pos in zip(batch,act,probes,batcharr,batchpositions):allf.append(pool.submit(execute,q,int(a),pb,ar,dispatch,pos,share))
     batch=[];batcharr=[];batchpositions=[];util.append(monitorutil())
   cellrows=[f.result() for f in allf]
  stop=max(r['completed_time'] for r in cellrows);span=stop-start;lat=np.array([r['arrival_to_completion_s'] for r in cellrows]);r=np.array([r['r'] for r in cellrows]);row={'domain':domain,'method':method,'window':W,'concurrency':conc,'load_index':load,'QPS_offered':qps,'request_n':N,'underlying_allowed_Dev_queries':len(queries),'mean_r':r.mean(),'harm_rate':np.mean(r>1e-10),'mean_latency_s':lat.mean(),'P50_s':np.quantile(lat,.5),'P95_s':np.quantile(lat,.95),'throughput_QPS':N/span,'goodput_QPS_latency_le200ms':int((lat<=.2).sum())/span,'latency_SLA_s':.2,'process_CPU_s':time.process_time()-processcpu,'batch_planning_CPU_s':pcpu,'batch_planning_wall_s':pwall,'paid_Probe_wall_s':probe,'mean_window_wait_s':np.mean([r['window_and_planning_wait_s'] for r in cellrows]),'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'NVML_SM_util_sample_mean_pct':np.mean([r['SM_util_pct'] for r in util if r['SM_util_pct'] is not None]),'GPU_activity_s':None,'GPU_activity_status':'secondary profile only; primary cell not instrumented','cohort':'PROFILED_SECONDARY' if profiled else 'UNPROFILED_PRIMARY','status':'EXPLORATORY_UNCERTIFIED_POLICY','Goodput_quality':'NO_CERTIFIED_QUALITY_GOODPUT'}
  return cellrows,row,util
 with torch.inference_mode():
  for p in plans:eng.execute(p,queries[0]['query'])
 configs=list(itertools.product(methodlist,[1,8,32],[1,2,4],[0,1,2]));random.Random(C.SEED).shuffle(configs)
 if mode=='aa':configs=[('Full-Visual',1,1,1)]
 for count,(method,W,conc,load) in enumerate(configs,1):
  guard.check();rows,stat,util=run_cell(method,W,conc,load);requests+=rows;cells.append(stat)
  with (C.A/'requests_partial.jsonl').open('a') as f:
   for r in rows:f.write(json.dumps(C.clean(r))+'\n')
  with (C.A/'cells_partial.jsonl').open('a') as f:f.write(json.dumps(C.clean(stat))+'\n')
  if count%5==0:print(json.dumps({'stage':'E4_STREAM','domain':domain,'completed_cells':count,'total':len(configs),'time_bj':C.stamp()}),flush=True)
 # Persist the primary cohort before any optional profiler replay can fail.
 C.write(C.A/'request_rows.json',requests);C.write(C.A/'cell_rows.json',cells);C.write(C.A/'setup.json',{'engine_setup':eng.setup,'cold_engine_load_wall_s':cold,'peak_host_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'CPU_affinity':sorted(os.sched_getaffinity(0))});C.table(C.R/'stream_cells.csv',cells)
 if mode=='main':
  for method in methodlist:
   guard.check();result,st=profile_call(lambda:run_cell(method,1,1,1,True),C.A/f'GPU_profile_{method}.json',require_kernels=method=='Full-Visual');profilerows.append({'domain':domain,'method':method,'window':1,'concurrency':1,'load_index':1,'requests':N,**st,'profiled_mean_arrival_latency_s':result[1]['mean_latency_s'],'primary_latency_used':False})
 C.write(C.A/'profile_rows.json',profilerows);print(json.dumps({'status':'PASS_E4_STREAM','domain':domain,'mode':mode,'cells':len(cells),'requests':len(requests),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
