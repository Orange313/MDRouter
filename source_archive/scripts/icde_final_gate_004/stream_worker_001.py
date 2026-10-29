import common as C
import engine_001 as E
import learning_base_001 as L
import os,sys,time,json,threading,random,resource,ctypes as ct,concurrent.futures,collections
import numpy as np
domain,gpu,phase,block_s,slot=sys.argv[1:];block=int(block_s);tag=f'{domain}_{phase}_b{block}_{slot}_001';C.start('S1/workers/'+tag)
spec=C.load(C.ART/'S1/streaming_protocol_v2.json');os.sched_setaffinity(0,spec['CPU_affinity'][str(gpu)]);E.idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json');stop=threading.Event();telemetry=[];state={'active':0,'inflight':0,'queue':0};state_lock=threading.Lock()
def hoststats():
 z={}
 with open('/proc/self/status') as f:
  for line in f:
   if line.startswith(('Threads:','VmRSS:','voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:')):k,v=line.split(':',1);z[k]=v.strip()
 with open('/proc/stat') as f:z['host_cpu_ticks']=list(map(int,f.readline().split()[1:]))
 with open('/proc/meminfo') as f:
  for line in f:
   if line.startswith('MemAvailable:'):z['MemAvailable_kB']=int(line.split()[1])
 z['loadavg']=os.getloadavg();ru=resource.getrusage(resource.RUSAGE_SELF);z.update(process_CPU_s=time.process_time(),voluntary_context_switches=ru.ru_nvcsw,involuntary_context_switches=ru.ru_nivcsw)
 return z
class Util(ct.Structure):_fields_=[('gpu',ct.c_uint),('memory',ct.c_uint)]
def observe():
 while not stop.wait(.5):
  try:
   u=Util();guard.lib.nvmlDeviceGetUtilizationRates(guard.handle,ct.byref(u));sm=ct.c_uint();mem=ct.c_uint();sc=guard.lib.nvmlDeviceGetClockInfo(guard.handle,1,ct.byref(sm));mc=guard.lib.nvmlDeviceGetClockInfo(guard.handle,2,ct.byref(mem))
   with state_lock:st=dict(state)
   telemetry.append({'monotonic':time.perf_counter(),'GPU_SM_util_pct':u.gpu,'GPU_memory_util_pct':u.memory,'GPU_SM_clock_MHz':sm.value if sc==0 else None,'GPU_mem_clock_MHz':mem.value if mc==0 else None,**st,**hoststats()})
  except Exception as ex:telemetry.append({'error':str(ex),'monotonic':time.perf_counter()})
monitor=threading.Thread(target=observe,daemon=True);monitor.start()
try:
 import torch
 plans=C.PLANS;models=C.load(C.ART/'S0/locked_models.json');md={m['method']:m for m in models};fitmodels={}
 for name,m in md.items():z=[np.array(x) for x in m['model']];z[0]=z[0].astype(bool);fitmodels[name]=z
 dc=next(x for x in C.load(C.ART/'S0/domain_policy_config.json') if x['domain']==domain);cost=np.array(dc['label_free_cost_vector'])
 if domain=='Pharmaceuticals':
  queries=[q for q in C.load(C.PREV/'E2/allowed_fitdev_queries.json') if q['role']=='development'];rr=C.load(C.PREV/'E2/outcomes_001/request_rows.json');expected={(int(r['query_id']),r['plan']):r['top10'] for r in rr if r['role']=='development'};rr=C.load(C.PREV/'E2/outcomes_001/outcomes.json');risk={(int(r['query_id']),r['plan']):r['r'] for r in rr if r['role']=='development'};groups={int(q['query_id']):q['family_id'] for q in queries};index='pharma_fixedcorpus_index_001'
 else:
  queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json');expected={(int(r['query_id']),r['plan']):r['top10'] for r in C.load(f'artifacts/icde_decisive_suite_002/R2/{domain}_candidate_rankings.json')};rr=C.load(C.PR/'E1/per_query_quality_cost.csv');risk={(int(r['query_id']),r['plan']):float(r['r']) for r in rr if r['domain']==domain};a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{domain}_dataset.npz');groups={int(q):g for q,g in zip(a['qids'],a['family_primary'])};index=dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']))[domain]
 t=time.perf_counter();eng=E.Engine(index,guard);torch.cuda.synchronize();cold=time.perf_counter()-t;pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='persistent_gpu');threadids=set();requests=[];cells=[]
 def invoke(q,method,rng,arrival,submitted,position,qps,cellid,warmup=False):
  start=time.perf_counter();threadids.add(threading.get_native_id());tc=time.thread_time()
  with state_lock:state['queue']-=1;state['active']+=1;initial=dict(state)
  guard.check();pt=time.perf_counter();pc=time.thread_time();pb=None;probe_wall=0.;pstat_wall=0.;qwall=0.;pred_wall=0.
  if method in plans:act=plans.index(method)
  elif method.startswith('RandomMix_'):act=int(rng.choice(4,p=dc['mixes'][method]))
  else:
   t=time.perf_counter();x=eng.Q(q['query']);qwall=time.perf_counter()-t
   if method=='QP_regret':
    t=time.perf_counter();pb=eng.bm.search(q['query']);probe_wall=time.perf_counter()-t;t=time.perf_counter();p=E.probe_features(eng.bm,q['query'],*pb);pstat_wall=time.perf_counter()-t;x=np.r_[x,p]
   t=time.perf_counter();prediction=L.predict(fitmodels[method],x[None])[0];utility=cost+md[method]['kappa']*np.maximum(prediction,0);utility[1]=np.inf;act=int(utility.argmin());pred_wall=time.perf_counter()-t
  planning_wall=time.perf_counter()-pt;planning_cpu=time.thread_time()-pc;path_start=time.perf_counter()
  with torch.inference_mode():top=eng.execute(plans[act],q['query'],pb);torch.cuda.synchronize()
  completed=time.perf_counter();cpu=time.thread_time()-tc;path_wall=completed-path_start;qid=int(q['query_id']);assert top==expected[qid,plans[act]],(domain,qid,method,plans[act],top,expected[qid,plans[act]])
  with state_lock:state['active']-=1;state['inflight']-=1
  return {'domain':domain,'phase':phase,'block':block,'slot':slot,'process_PID':os.getpid(),'native_thread':threading.get_native_id(),'GPU_physical':int(gpu),'method':method,'cell_id':cellid,'position':position,'query_id':qid,'family':groups[qid],'plan':plans[act],'r':risk[qid,plans[act]],'QPS_offered':qps,'scheduled_arrival_ts':arrival,'submitted_ts':submitted,'start_ts':start,'complete_ts':completed,'arrival_to_completion_s':completed-arrival,'service_wall_s':completed-start,'executor_queue_s':start-submitted,'arrival_dispatch_lateness_s':submitted-arrival,'service_thread_CPU_s':cpu,'planning_CPU_s':planning_cpu,'planning_wall_s':planning_wall,'Q_feature_wall_s':qwall,'Probe_wall_s':probe_wall,'Pstats_wall_s':pstat_wall,'predict_allocate_wall_s':pred_wall,'retrieval_path_wall_s':path_wall,'Probe_reused':pb is not None and act!=0,'Probe_paid_before_FullV':pb is not None and act==0,'active_worker':initial['active'],'inflight_at_start':initial['inflight'],'queue_depth_at_start':initial['queue'],'timeout':False,'warmup':warmup,'cohort':'UNPROFILED_PRIMARY','status':'DEV_EXPLORATORY_NOT_CERTIFIED'}
 def submit(q,method,rng,ar,pos,qps,cid,warm=False):
  with state_lock:state['inflight']+=1;state['queue']+=1
  sub=time.perf_counter();return pool.submit(invoke,q,method,rng,ar,sub,pos,qps,cid,warm)
 rng=np.random.default_rng(C.SEED)
 for i in range(spec['warmup']['per_fresh_process']):submit(queries[i%len(queries)],'Full-Visual',rng,time.perf_counter(),i,0,'warmup',True).result()
 def cell(method,qps,li):
  # Actual worker-thread warmups are explicit and occur BEFORE every measurement schedule.
  for i in range(4):submit(queries[i],method,rng,time.perf_counter(),i,0,'intercell_warmup',True).result()
  ordered=queries.copy();random.Random(C.SEED+block*100+li).shuffle(ordered);work=ordered+ordered;rngcell=np.random.default_rng(C.SEED+block*100+li+sum(map(ord,method)));cid=f'{method}_{li}';start=time.perf_counter()+.05;cpu0=time.process_time();h0=hoststats();futures=[]
  for i,q in enumerate(work):
   ar=start+i/qps;delay=ar-time.perf_counter()
   if delay>0:time.sleep(delay)
   futures.append(submit(q,method,rngcell,ar,i,qps,cid))
  rs=[f.result() for f in futures];end=max(r['complete_ts'] for r in rs);lat=np.array([r['arrival_to_completion_s'] for r in rs]);service=np.array([r['service_wall_s'] for r in rs]);h1=hoststats()
  row={'domain':domain,'phase':phase,'block':block,'slot':slot,'GPU':int(gpu),'PID':os.getpid(),'method':method,'load_index':li,'QPS_offered':qps,'requests':len(rs),'distinct_query_n':len(queries),'mean_service_wall_s':service.mean(),'mean_arrival_to_completion_s':lat.mean(),'P50_s':np.quantile(lat,.5),'P95_s':np.quantile(lat,.95),'P99_s':np.quantile(lat,.99),'mean_queue_s':np.mean([r['executor_queue_s'] for r in rs]),'mean_arrival_dispatch_lateness_s':np.mean([r['arrival_dispatch_lateness_s'] for r in rs]),'completed_QPS':len(rs)/(end-start),'arrival_limited_throughput_note':'includes idle inter-arrival span; not maximum sustainable throughput','goodput_QPS_le200ms':np.sum(lat<=.2)/(end-start),'mean_r':np.mean([r['r'] for r in rs]),'process_CPU_s':time.process_time()-cpu0,'mean_thread_CPU_s':np.mean([r['service_thread_CPU_s'] for r in rs]),'voluntary_context_switches_delta':h1['voluntary_context_switches']-h0['voluntary_context_switches'],'involuntary_context_switches_delta':h1['involuntary_context_switches']-h0['involuntary_context_switches'],'warmup_count_predeclared':4,'timeout_n':0,'max_inflight':max(r['inflight_at_start'] for r in rs),'max_queue_depth_at_start':max(r['queue_depth_at_start'] for r in rs),'GPU_activity_s':None,'GPU_scope':'unprofiled primary; separate old/new exact-path profiles only, not imputed','status':'MEASURED_NOT_CERTIFIED'}
  requests.extend(rs);cells.append(row);print(json.dumps({'tag':tag,'cell':cid,'QPS':qps,'requests':len(rs),'mean_latency_s':lat.mean(),'service_s':service.mean(),'time_bj':C.stamp()}),flush=True)
 if phase=='light':
  loads=[(0,2.),(1,5.)];loads=loads if block==0 else loads[::-1]
  for li,qps in loads:cell('Full-Visual',qps,li)
 elif phase=='stress':
  loads=[(2,.65/dc['FullV_resident_mean_s']),(3,.95/dc['FullV_resident_mean_s'])];loads=loads if slot in ['A1','B1'] else loads[::-1]
  for li,qps in loads:cell('Full-Visual',qps,li)
 elif phase=='routing':
  methods=spec['routing_after_gate']['methods'].copy();random.Random(C.SEED).shuffle(methods);methods=methods[block:]+methods[:block]
  for li,qps in [(0,2.),(1,5.)]:
   for method in methods:cell(method,qps,li)
 else:raise ValueError(phase)
 pool.shutdown(wait=True);assert len(threadids)==1;C.write(C.A/'request_rows.json',requests);C.write(C.A/'cell_rows.json',cells);C.write(C.A/'setup.json',{'engine_setup':eng.setup,'cold_wall_s':cold,'CPU_affinity':sorted(os.sched_getaffinity(0)),'worker_thread_ids':list(threadids),'torch_num_threads':torch.get_num_threads(),'peak_host_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'pid':os.getpid(),'pool_rebuilt_per_cell':False});C.table(C.R/'cells.csv',cells)
finally:
 stop.set();monitor.join(timeout=2);C.write(C.A/'telemetry.json',telemetry);guard.close()
print(json.dumps({'tag':tag,'status':'PASS_WORKER_COMPLETE','time_bj':C.stamp()}),flush=True)
