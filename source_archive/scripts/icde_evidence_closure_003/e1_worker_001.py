import common as C
import numpy as np,os,sys,time,json,subprocess,collections,ast,math,random,shutil,resource
domain,rep,gpu=sys.argv[1:];rep=int(rep);tag=f'{domain}_process_{rep:02d}'
C.start('E1/workers/'+tag);C.STAGE='E1_'+tag
idle=[]
for i in range(3):
 raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();z=raw.split(',');ok=int(z[2].split()[0])<128 and int(z[3].split()[0])==0;idle.append({'time_bj':C.stamp(),'raw':raw,'idle':ok})
 if not ok:C.write(C.A/'three_idle_checks.json',idle);raise RuntimeError('GPU not idle; no CUDA context started')
 time.sleep(1)
C.write(C.A/'three_idle_checks.json',idle)
assert shutil.disk_usage(C.ROOT).free>15*2**30,'Storage reserve'
cores=sorted(os.sched_getaffinity(0));chosen=cores[(int(gpu)*3)%len(cores):][:2];os.sched_setaffinity(0,chosen)
C.write(C.A/'CPU_host_context.json',{'affinity':chosen,'load':os.getloadavg(),'isolation':'GPU exclusive via ResourceGuard; host shared'})
sys.path.insert(0,str(C.ROOT/'src'))
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import torch
 from transformers import AutoModel,AutoTokenizer
 from colpali_engine.models import ColQwen2,ColQwen2Processor
 from aaber.common.text import tokenize
 from mdrouter_v5.graded_metrics import stable_ranking
 torch.set_num_threads(1);torch.manual_seed(C.SEED+rep);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ip=C.ROOT/'cache/indexes'/dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']))[domain]
 models=C.load('manifests/hr_models_verified.json','FROZEN_MODEL_METADATA');source=C.load('scripts/cross_domain_four_plan_001.py','FROZEN_BACKEND_CODE');cfg=C.load('configs/hr_pilot_sparse_004.json','FROZEN_SCORE_SEMANTICS')
 nodes=[x for x in ast.parse(source).body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ['BM25','load_visual','load_dense','encode_dense']]
 env={'MODELS':models,'CFG':cfg,'torch':torch,'AutoModel':AutoModel,'AutoTokenizer':AutoTokenizer,'ColQwen2':ColQwen2,'ColQwen2Processor':ColQwen2Processor,'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'frozen_backend_AST','exec'),env)
 setup={};t0=time.perf_counter();c0=time.process_time();pages=C.load(ip/'pages.json','UNLABELED_SHARED_CORPUS');ids=[int(x['corpus_id']) for x in pages]
 t=time.perf_counter();ct=time.process_time();bm=env['BM25'](pages);setup.update(BM25_build_wall_s=time.perf_counter()-t,BM25_build_CPU_s=time.process_time()-ct)
 t=time.perf_counter();model,processor=env['load_visual']();torch.cuda.synchronize();setup['visual_model_load_wall_s']=time.perf_counter()-t
 t=time.perf_counter();visual={};im=C.load(ip/'index_manifest.json','FROZEN_INDEX_METADATA')
 for path in sorted((ip/'visual_parts').glob('*.pt')):
  guard.check();actual=C.sha(path);assert actual==im['files'][str(path.relative_to(ip))];C.INPUT[str(path.relative_to(C.ROOT))]={'sha256':actual,'role':'FROZEN_UNLABELED_VISUAL_INDEX'};part=torch.load(path,weights_only=True);visual.update(zip(part['ids'],part['embeddings']))
 passage=torch.nn.utils.rnn.pad_sequence([visual[p].float() for p in ids],batch_first=True).cuda();del visual;torch.cuda.synchronize();setup['visual_index_residency_wall_s']=time.perf_counter()-t
 t=time.perf_counter();dm,dt=env['load_dense']();dense=torch.load(ip/'dense.pt',weights_only=True);dense_emb=dense['embeddings'].cuda();chunk_pages=dense['chunk_pages'].cuda();torch.cuda.synchronize();setup['dense_model_index_load_wall_s']=time.perf_counter()-t;C.INPUT[str((ip/'dense.pt').relative_to(C.ROOT))]={'sha256':C.sha(ip/'dense.pt'),'role':'FROZEN_DENSE_INDEX'}
 setup.update(total_joint_cold_setup_wall_s=time.perf_counter()-t0,total_joint_cold_setup_CPU_s=time.process_time()-c0)
 queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json','ALLOWED_DEV_REQUESTS')
 expected={(int(x['query_id']),x['plan']):x['top10'] for x in C.load(f'artifacts/icde_decisive_suite_002/R2/{domain}_candidate_rankings.json','DEV_EXACT_REPLAY_EXPECTED_OUTPUTS')}
 def encode(q):return model(**processor.process_queries([q]).to('cuda:0'))[0].float()
 def score(q,indices):
  if not indices:return []
  out=[]
  for start in range(0,len(indices),64):
   subset=indices[start:start+64];block=passage.index_select(0,torch.tensor(subset,device='cuda:0'))
   if len(subset)<64:block=torch.nn.functional.pad(block,(0,0,0,0,0,64-len(subset)))
   vals=torch.matmul(q.unsqueeze(0).expand(64,-1,-1),block.transpose(1,2)).max(2).values.clamp_min(0).sum(1);out.append(vals[:len(subset)])
  vv=torch.cat(out).cpu().tolist();return [indices[j] for j in stable_ranking([ids[j] for j in indices],vv)]
 def dense_order(text):
  q=env['encode_dense'](dm,dt,texts=[cfg['dense']['query_prefix']+text])[0];vv=dense_emb@q;s=torch.full((len(ids),),-float('inf'),device='cuda:0');s.scatter_reduce_(0,chunk_pages,vv,reduce='amax',include_self=True);return stable_ranking(ids,s.cpu().tolist())
 def execute(plan,text):
  if plan=='BM25':return [ids[i] for i in bm.search(text)[0]][:10]
  if plan=='Full-Visual':cand=list(range(len(ids)))
  elif plan.startswith('BM25-R'):cand=bm.search(text,int(plan[6:]))[0]
  elif plan=='Dense-R50':cand=dense_order(text)[:50]
  elif plan=='Hybrid-R50':
   br=bm.search(text)[0];dr=dense_order(text);bp={p:i+1 for i,p in enumerate(br)};dp={p:i+1 for i,p in enumerate(dr)};pool=set(br[:50])|set(dr[:50]);cand=sorted(pool,key=lambda p:(-(1/(60+bp[p]) if p in bp else 0)-(1/(60+dp[p]) if p in dp else 0),ids[p]))[:50]
  else:raise ValueError(plan)
  if not cand:return []
  return [ids[i] for i in score(encode(text),cand)][:10]
 def checked(plan,q):
  actual=execute(plan,q['query']);assert actual==expected[int(q['query_id']),plan],(domain,plan,q['query_id'],'Top10 mismatch',actual,expected[int(q['query_id']),plan]);return actual
 plans=['BM25','BM25-R50','BM25-R100','BM25-R200','Dense-R50','Hybrid-R50','Full-Visual'];rows=[];profiles=[];components=[]
 def append(name,row):
  with (C.A/name).open('a') as f:f.write(json.dumps(C.clean(row))+'\n');f.flush()
 with torch.inference_mode():
  for p in plans:checked(p,queries[0])
  for repeat in range(3):
   work=queries.copy();random.Random(C.SEED+rep*1009+repeat).shuffle(work)
   for i,q in enumerate(work):
    pp=plans.copy();random.Random(C.SEED+rep*1009+repeat*101+i).shuffle(pp)
    for plan in pp:
     guard.check();torch.cuda.synchronize();ct=time.process_time();t=time.perf_counter();result=checked(plan,q);torch.cuda.synchronize();row={'domain':domain,'query_id':int(q['query_id']),'process':rep,'repeat':repeat,'position':i,'plan':plan,'CPU_s':time.process_time()-ct,'resident_E2E_s':time.perf_counter()-t,'top10':result,'status':'PASS_EXACT_OUTPUT','Probe_reused_within_path':plan.startswith(('BM25','Hybrid')),'cache':'resident; no cached query encodings'};rows.append(row);append('request_partial.jsonl',row)
   print(json.dumps({'worker':tag,'time_bj':C.stamp(),'completed_repeat':repeat,'requests':len(rows)}),flush=True)
  work=queries.copy();random.Random(C.SEED+rep*1013).shuffle(work)
  for i,q in enumerate(work):
   for plan in random.Random(C.SEED+rep+i).sample(plans,len(plans)):
    guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30,'Storage reserve';trace=C.A/f'trace_{q["query_id"]}_{plan}.json';_,st=profile_call(lambda:checked(plan,q),trace,require_kernels=plan!='BM25');row={'domain':domain,'query_id':int(q['query_id']),'process':rep,'plan':plan,**st,'trace':str(trace.relative_to(C.ROOT)),'cohort':'separate GPU profiling; not E2E sample'};profiles.append(row);append('profile_partial.jsonl',row)
   if (i+1)%10==0:print(json.dumps({'worker':tag,'time_bj':C.stamp(),'profile_queries':i+1}),flush=True)
  # Separate component calls: each query, one process; never force separate profiles to sum.
  if rep==0:
   for q in queries:
    encoded,st=profile_call(lambda:encode(q['query']),C.A/f'component_encode_{q["query_id"]}.json');components.append({'domain':domain,'query_id':int(q['query_id']),'component':'visual_query_encoding','GPU_activity_s':st['gpu_active_union_seconds']})
    for k in [50,100,200,len(ids)]:
     cand=bm.search(q['query'],k)[0] if k<len(ids) else list(range(len(ids)));_,st=profile_call(lambda:score(encoded,cand),C.A/f'component_score_{q["query_id"]}_{k}.json');components.append({'domain':domain,'query_id':int(q['query_id']),'component':f'visual_score_{k}','GPU_activity_s':st['gpu_active_union_seconds']})
 setup.update(peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_affinity=chosen)
 C.write(C.A/'setup.json',setup);C.write(C.A/'request_rows.json',rows);C.write(C.A/'profile_rows.json',profiles);C.write(C.A/'component_rows.json',components);C.table(C.R/'request_latencies.csv',rows);C.table(C.R/'GPU_activity_profiles.csv',profiles);C.table(C.R/'GPU_component_profiles.csv',components)
 print(json.dumps({'worker':tag,'status':'PASS','requests':len(rows),'profiles':len(profiles),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
