import common as C
import subprocess,os,time,pathlib,json
domain='vidore_v3_finance_en';gpu=1;phase='light';runs=[]
C.write(C.ART/'S1/finance_resume_registration_002.json',{'time_bj':C.stamp(),'reason':'attempt001 aborted BEFORE CUDA and BEFORE any measurement: idle check GPU1 15MiB, utilization1%; no outcome-driven rerun','same_scientific_protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_002.py'),'all_attempt001_failures_preserved':True,'attempt':'002','no_optional_repeat_after_measurement':True})
for b in range(2):
 for slot in ['A1','B1','B2','A2']:
  # Require three consecutive quiet preflight readings, then worker repeats its own three checks.
  quiet=0
  while quiet<3:
   raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True);a,z=raw.split(',');quiet=quiet+1 if int(a.split()[0])<128 and int(z.split()[0])==0 else 0;time.sleep(1)
  tag=f'{domain}_light_b{b}_{slot}_002';log=C.ART/'logs'/f'{tag}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_002.py'),domain,'1','light',str(b),slot],stdout=f,stderr=subprocess.STDOUT,cwd=C.ROOT,env=env)
  runs.append({'tag':tag,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(runs[-1]),flush=True)
  if r.returncode:raise RuntimeError('Resource/worker failure retained; stop affected domain')
C.write(C.ART/'S1/finance_resume_completion_002.json',{'status':'PASS_COMPLETED','runs':runs,'time_bj':C.stamp()})
