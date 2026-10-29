import common as C
import subprocess,time,pathlib,json,os
pending={d:[(b,s) for b in range(2) for s in ['A1','B1','B2','A2'] if not (C.ART/f'S1/workers/{d}_light_b{b}_{s}_001/cell_rows.json').exists()] for d in ['vidore_v3_hr','Pharmaceuticals']}
# Both failures were initial idle-check residual1%, no CUDA, no requests. Keep ALL completed processes.
for d in pending:
 for b,s in pending[d]:
  p=C.ART/f'S1/workers/{d}_light_b{b}_{s}_001'
  if (p/'failure.json').exists():assert not (p/'request_rows.json').exists() and not (p/'resource_guard.json').exists()
C.write(C.ART/'S1/resource_resume_registration_003.json',{'time_bj':C.stamp(),'missing_scheduled_processes':pending,'reason':'initial three-idle-check utilization1% before CUDA; no measured observations discarded','all_existing_successes_retained':True,'protocol_unchanged_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'worker_code_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_003.py'),'ABBA_interruption':'resource preflight interrupts blocks; retain elapsed-time gaps and all processes; no result-based selection','rerun_measurements':False})
runs=[]
for d,items in pending.items():
 for b,slot in items:
  quiet=0
  while quiet<3:
   raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True);m,u=raw.split(',');quiet=quiet+1 if int(m.split()[0])<128 and int(u.split()[0])==0 else 0;time.sleep(1)
  tag=f'{d}_light_b{b}_{slot}_002';log=C.ART/'logs'/f'{tag}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_003.py'),d,'0','light',str(b),slot,'002'],stdout=f,stderr=subprocess.STDOUT,cwd=C.ROOT,env=env)
  runs.append({'domain':d,'tag':tag,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(runs[-1]),flush=True)
  if r.returncode:raise RuntimeError('Retain failure and stop affected lane, no post-result repeat')
C.write(C.ART/'S1/resource_resume_completion_003.json',{'status':'PASS_COMPLETED','runs':runs,'time_bj':C.stamp()})
