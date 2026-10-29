import common as C
import pathlib,json,subprocess,os,time,concurrent.futures
def complete(p):
 return (p/'cell_rows.json').exists() and (p/'resource_guard.json').exists() and json.loads((p/'resource_guard.json').read_text())['valid']
selected={};pending={d:[] for d in C.DOM};invalid=[]
for d in C.DOM:
 for b in range(2):
  for s in ['A1','B1','B2','A2']:
   roots=sorted((C.ART/'S1/workers').glob(f'{d}_light_b{b}_{s}_*'));valid=[p for p in roots if complete(p)];assert len(valid)<=1,'No repeated healthy measurements allowed'
   if valid:selected[f'{d}|{b}|{s}']=str(valid[0].relative_to(C.ROOT))
   else:pending[d].append((b,s))
   for p in roots:
    if not complete(p):invalid.append({'path':str(p),'failure':json.loads((p/'failure.json').read_text())['exception'] if (p/'failure.json').exists() else 'NO_COMPLETE_PROCESS','retain':True})
reg={'time_bj':C.stamp(),'valid_processes_retained':selected,'missing_processes':pending,'invalid_attempts_retained':invalid,'retry_basis':'only pre-CUDA strict idle failure or ResourceGuard foreign-PID invalidation; no valid process removed or repeated; no threshold/seed/load/model change','idle_amendment':'automatic waiting until SAME 3 strict idle checks, rather than terminating initial busy sample; no CUDA while waiting','trace_amendment':'save cell logs outside measured intervals, plus all completed invoke rows in finally for future failed attempts','protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_004.py')};C.write(C.ART/'S1/resource_recovery_registration_005.json',reg)
assign={'vidore_v3_hr':0,'vidore_v3_finance_en':1,'vidore_v3_industrial':4,'Pharmaceuticals':0}
def lane(gpu):
 out=[]
 for d in C.DOM:
  if assign[d]!=gpu:continue
  for b,s in pending[d]:
   tag=f'{d}_light_b{b}_{s}_005';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');log=C.ART/'logs'/f'{tag}.log'
   with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_004.py'),d,str(gpu),'light',str(b),s,'005'],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=C.ROOT)
   rec={'domain':d,'tag':tag,'returncode':r.returncode,'time_bj':C.stamp()};out.append(rec);print(json.dumps(rec),flush=True)
   if r.returncode:return out
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rr=[x for r in ex.map(lane,[0,1,4]) for x in r]
C.write(C.ART/'S1/resource_recovery_completion_005.json',{'runs':rr,'time_bj':C.stamp(),'all_completed':all(x['returncode']==0 for x in rr)})
