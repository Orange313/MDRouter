import common as C
import pathlib,json,subprocess,os
d='Pharmaceuticals';pending=[]
for b in range(2):
 for s in ['A1','B1','B2','A2']:
  valid=[]
  for p in (C.ART/'S1/workers').glob(f'{d}_light_b{b}_{s}_*'):
   if (p/'cell_rows.json').exists() and (p/'resource_guard.json').exists() and json.loads((p/'resource_guard.json').read_text())['valid']:valid.append(p)
  if not valid:pending.append((b,s))
C.write(C.ART/'S1/pharma_resource_resume_registration_007.json',{'time_bj':C.stamp(),'pending':pending,'reason':'foreign GPU PID1527619 invalidated prior attempt; no performance-based repeats; raw partial/ResourceGuard retained','same_protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_004.py'),'resource_failure_branch':'if foreign PID recurs stop affected domain INSUFFICIENT_RESOURCE; no scanning for favorable data'})
rr=[]
for b,s in pending:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');tag=f'{d}_light_b{b}_{s}_007';log=C.ART/'logs'/f'{tag}.log'
 with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_004.py'),d,'0','light',str(b),s,'007'],cwd=C.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 rr.append({'tag':tag,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(rr[-1]),flush=True)
 if r.returncode:break
C.write(C.ART/'S1/pharma_resource_resume_completion_007.json',{'runs':rr,'time_bj':C.stamp(),'all_complete':len(rr)==len(pending) and all(r['returncode']==0 for r in rr)})
