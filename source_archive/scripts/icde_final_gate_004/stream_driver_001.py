import common as C
import subprocess,concurrent.futures,sys,json,time,pathlib,os
phase=sys.argv[1];C.A=C.ART/'S1';C.R=C.REPORT/'S1';C.STAGE='S1';sys.addaudithook(C.protected)
assert (C.ART/'S0/asset_binding_001/manifest.json').exists(),'Asset binding must complete before GPU'
assert json.loads((C.ART/'S0/asset_binding_001/manifest.json').read_text())['status']=='PASS'
spec=C.load(C.A/'streaming_protocol_v2.json');bind=json.loads((C.A/'execution_code_lock_001.json').read_text())
for p,h in bind['code_SHA256'].items():assert C.sha(C.ROOT/p)==h,(p,'code changed after pre-run lock')
domains=C.DOM
if phase!='light':
 gate=C.load(C.A/'AA_gate.json');domains=[d for d in domains if gate['domains'][d]['status']=='PASS'];assert domains,'No domains passed AA; dependent stream forbidden'
def one_domain(domain,gpu):
 results=[]
 for block,slot in ([(b,s) for b in range(2) for s in ['A1','B1','B2','A2']] if phase=='light' else [(0,s) for s in ['A1','B1','B2','A2']] if phase=='stress' else [(b,'R') for b in range(3)]):
  # Wait only for the prescribed physical GPU; never kill or evict other processes.
  while True:
   raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True);a,b=raw.split(',')
   if int(a.split()[0])<128 and int(b.split()[0])==0:break
   print(json.dumps({'status':'WAIT_IDLE','gpu':gpu,'domain':domain,'time_bj':C.stamp()}),flush=True);time.sleep(10)
  tag=f'{domain}_{phase}_b{block}_{slot}_001';log=C.ART/'logs'/f'{tag}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_001.py'),domain,str(gpu),phase,str(block),slot],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=C.ROOT)
  results.append({'domain':domain,'gpu':gpu,'tag':tag,'returncode':r.returncode,'log':str(log)})
  print(json.dumps({**results[-1],'time_bj':C.stamp()}),flush=True)
  if r.returncode:break
 return results
# Fixed hardware for each domain across all independent processes and phases.
assign={'vidore_v3_hr':0,'vidore_v3_finance_en':1,'vidore_v3_industrial':4,'Pharmaceuticals':0}
def lane(gpu):
 out=[]
 for d in domains:
  if assign[d]==gpu:out.extend(one_domain(d,gpu))
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:allrows=[r for rs in ex.map(lane,[0,1,4]) for r in rs]
C.write(C.A/f'{phase}_driver_completion.json',{'time_bj':C.stamp(),'phase':phase,'runs':allrows,'expected':len(domains)*(8 if phase=='light' else 4 if phase=='stress' else 3),'failed':sum(x['returncode']!=0 for x in allrows)});print(json.dumps({'phase':phase,'completed':len(allrows),'failed':sum(x['returncode']!=0 for x in allrows),'time_bj':C.stamp()}),flush=True)
