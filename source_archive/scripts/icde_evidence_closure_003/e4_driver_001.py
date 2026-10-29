import common as C
import subprocess,os,time,json,sys,pathlib
C.A=C.ART/'E4';C.R=C.REPORT/'E4';assert (C.A/'prestream_asset_lock.json').exists()
pending=[{'domain':d,'mode':'main','forced_gpu':None} for d in ['Pharmaceuticals']+C.DOM];running={};done=[];errors=[];last=0
while pending or running:
 for gpu,j in list(running.items()):
  code=j['proc'].poll()
  if code is None:continue
  j['log'].close();del running[gpu];record={k:v for k,v in j.items() if k not in ['proc','log']};record['exit_code']=code;record['finished_time_bj']=C.stamp();done.append(record)
  if code:errors.append(record)
  if j['mode']=='main':
   worker=C.A/'workers'/f'{j["domain"]}_main_001';guard=worker/'resource_guard.json'
   if code==0 or ((worker/'cell_rows.json').exists() and guard.exists() and json.loads(guard.read_text()).get('valid')):pending.insert(0,{'domain':j['domain'],'mode':'aa','forced_gpu':gpu})
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True);free=[]
 for line in raw.strip().splitlines():
  i,mem,util=line.split(',');i=i.strip()
  if int(mem.split()[0])<128 and int(util.split()[0])==0 and i not in running:free.append(i)
 for gpu in free:
  if len(running)>=3:break
  ix=next((i for i,j in enumerate(pending) if j['forced_gpu'] in [None,gpu]),None)
  if ix is None:continue
  j=pending.pop(ix);log=(C.A/f'{j["domain"]}_{j["mode"]}_driver_001.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false');cmd=[sys.executable,str(C.ROOT/f'scripts/{C.RUN}/e4_worker_001.py'),j['domain'],gpu,j['mode']];p=subprocess.Popen(cmd,cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);running[gpu]={**j,'gpu':gpu,'pid':p.pid,'start_time_bj':C.stamp(),'proc':p,'log':log};print(json.dumps({'status':'STARTED','domain':j['domain'],'mode':j['mode'],'GPU':gpu,'pid':p.pid,'time_bj':C.stamp()}),flush=True)
 if time.monotonic()-last>60:print(json.dumps({'time_bj':C.stamp(),'running':[{'gpu':g,'domain':j['domain'],'mode':j['mode']} for g,j in running.items()],'pending':pending}),flush=True);last=time.monotonic()
 time.sleep(3)
C.write(C.A/'driver_completion.json',{'time_bj':C.stamp(),'completed_jobs':done,'errors':errors,'all_successful':not errors});print(json.dumps({'status':'PASS' if not errors else 'PARTIAL_WITH_FAILURES','jobs':len(done),'time_bj':C.stamp()}),flush=True)
