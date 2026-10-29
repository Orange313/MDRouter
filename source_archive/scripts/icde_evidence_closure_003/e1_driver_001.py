import common as C
import subprocess,os,time,json,sys
C.start('E1');jobs=[];assign=dict(zip(C.DOM,sys.argv[1:]));assert len(assign)==3
C.write(C.A/'driver_protocol.json',{'time_bj':C.stamp(),'GPU_assignment':assign,'processes_per_domain':3,'repeats_per_process':3,'expected_unprofiled_requests':7497,'expected_profiled_requests':2499})
for domain,gpu in assign.items():
 jobs.append({'domain':domain,'gpu':gpu,'next':0,'proc':None,'failed':False})
while any(j['next']<3 or j['proc'] is not None for j in jobs if not j['failed']):
 for j in jobs:
  if j['failed']:continue
  if j['proc'] is not None:
   code=j['proc'].poll()
   if code is None:continue
   j['log'].close();j['proc']=None
   if code!=0:j['failed']=True;print(json.dumps({'domain':j['domain'],'status':'FAILED_PRESERVED','exit':code,'time_bj':C.stamp()}),flush=True);continue
  if j['next']<3:
   r=j['next'];log=(C.A/f'{j["domain"]}_process_{r:02d}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=j['gpu'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false');cmd=[sys.executable,str(C.ROOT/f'scripts/{C.RUN}/e1_worker_001.py'),j['domain'],str(r),j['gpu']];j['proc']=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=C.ROOT);j['log']=log;j['next']+=1;print(json.dumps({'domain':j['domain'],'process':r,'physical_GPU':j['gpu'],'pid':j['proc'].pid,'status':'STARTED','time_bj':C.stamp()}),flush=True)
 time.sleep(3)
C.write(C.A/'driver_completion.json',{'jobs':[{k:v for k,v in j.items() if k not in ['proc','log']} for j in jobs],'time_bj':C.stamp(),'all_successful':not any(j['failed'] for j in jobs)})
if any(j['failed'] for j in jobs):raise RuntimeError('E1 worker failure; artifacts preserved, affected claims withheld')
print('ALL_E1_WORKERS_COMPLETED',flush=True)
