import common as C,subprocess,time,json,os,pathlib,shutil
from scipy.stats import t
import numpy as np
C.A=C.ART/'A2';C.R=C.REPORT/'A2';spec=C.load(C.A/'streaming_protocol_v3.json');deadline=time.monotonic()+60*spec['resource_plan']['max_idle_wait_minutes'];logs=C.ART/'logs';schedule=[];occupied=[];gates=[]
def idle_gpu(fixed=None):
 while time.monotonic()<deadline:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);rows=[[int(v.strip()) for v in x.split(',')] for x in raw.strip().splitlines()];record={'time_bj':C.stamp(),'GPU_rows':rows};occupied.append(record)
  with (C.A/'resource_monitor.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  for g,mem,u in rows:
   if g in spec['resource_plan']['allowed_GPUs'] and (fixed is None or g==fixed) and mem<128 and u==0:return g
  time.sleep(10)
 return None
def run(d,g,phase,b,s):
 tag=f'{d}_{phase}_b{b}_{s}_001';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 assert shutil.disk_usage(C.ROOT).free>=15*1024**3
 with (logs/(tag+'.log')).open('x') as f:z=subprocess.run([str(C.ROOT/'.venv/bin/python'),str(pathlib.Path(__file__).parent/'stream_worker_v3_001.py'),d,str(g),phase,str(b),s,'001'],env=env,stdout=f,stderr=subprocess.STDOUT)
 p=C.A/('policy_workers' if phase=='routing' else 'workers')/tag;ok=z.returncode==0 and (p/'resource_guard.json').exists() and json.loads((p/'resource_guard.json').read_text())['valid'] and (p/'cell_rows.json').exists();schedule.append({'domain':d,'phase':phase,'block':b,'slot':s,'GPU':g,'path':str(p),'returncode':z.returncode,'status':'PASS_WORKER_COMPLETE' if ok else 'ABORTED_RETAINED'});print(json.dumps(schedule[-1]),flush=True);return ok,p
def gate(d,paths):
 vals=[]
 for b in range(2):
  rows={s:json.loads((paths[b,s]/'cell_rows.json').read_text()) for s in ['A1','B1','B2','A2']}
  for q in [2.,5.]:
   for metric in ['mean_service_wall_s','mean_arrival_to_completion_s']:
    for a,z in [('A1','B1'),('A2','B2')]:
     av=next(x[metric] for x in rows[a] if x['QPS_offered']==q);bv=next(x[metric] for x in rows[z] if x['QPS_offered']==q);vals.append(dict(QPS=q,metric=metric,logratio=np.log(bv/av)))
 checks=[]
 for q in [2.,5.]:
  for metric in ['mean_service_wall_s','mean_arrival_to_completion_s']:
   v=np.array([x['logratio'] for x in vals if x['QPS']==q and x['metric']==metric]);mu=v.mean();hw=t.ppf(1-.0125/2,3)*v.std(ddof=1)/2;lo,hi=np.exp([mu-hw,mu+hw]);checks.append(dict(domain=d,QPS=q,metric=metric,ratio=np.exp(mu),CI_low=lo,CI_high=hi,status='PASS' if lo>=.9 and hi<=1.1 else 'INSUFFICIENT_EQUIVALENCE',paired_process_n=4,CI_level=.9875))
 C.write(C.A/f'{d}_AA_gate.json',{'domain':d,'checks':checks,'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'INSUFFICIENT_EQUIVALENCE','time_bj':C.stamp()});gates.extend(checks);return all(x['status']=='PASS' for x in checks)
for d in spec['domains']:
 paths={};bad=False;fixed=None
 for b in range(2):
  for s in ['A1','B1','B2','A2']:
   g=idle_gpu(fixed)
   if g is None:schedule.append(dict(domain=d,phase='light',block=b,slot=s,status='BLOCKED_NO_IDLE_GPU'));bad=True;break
   fixed=g;ok,p=run(d,g,'light',b,s);paths[b,s]=p
   if not ok:bad=True;break
  if bad:break
 if bad:continue
 if gate(d,paths):
  for b in range(3):
   g=idle_gpu(fixed)
   if g is None:schedule.append(dict(domain=d,phase='routing',block=b,status='BLOCKED_NO_IDLE_GPU'));break
   ok,p=run(d,g,'routing',b,'R')
   if not ok:break
C.table(C.R/'execution_schedule.csv',schedule);C.table(C.R/'aa_equivalence_v3.csv',gates,fields=['domain','QPS','metric','ratio','CI_low','CI_high','status','paired_process_n','CI_level']);C.write(C.A/'driver_complete.json',{'time_bj':C.stamp(),'status':'EXECUTION_FINISHED_WITH_EXPLICIT_GATES','schedule':schedule,'AA_checks':gates,'resource_polls':len(occupied)});print(json.dumps({'status':'DRIVER_FINISHED','time_bj':C.stamp(),'schedule_n':len(schedule),'AA_checks':gates}),flush=True)
