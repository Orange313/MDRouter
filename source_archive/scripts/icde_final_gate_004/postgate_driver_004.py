import common as C
import sys,subprocess,os,pathlib,time,json
d,gpu=sys.argv[1:];gate=json.loads((C.ART/f'S1/AA_gate_{d}_001.json').read_text());assert gate['status']=='PASS','AA failed; no stress or routing allowed';runs=[]
C.write(C.ART/f'S1/postgate_code_lock_{d}_001.json',{'time_bj':C.stamp(),'gate_SHA256':C.sha(C.ART/f'S1/AA_gate_{d}_001.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_003.py'),'same_protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json')})
for phase,configs in [('stress',[(0,s) for s in ['A1','B1','B2','A2']]),('routing',[(b,'R') for b in range(3)])]:
 for b,s in configs:
  quiet=0
  while quiet<3:
   raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True);m,u=raw.split(',');quiet=quiet+1 if int(m.split()[0])<128 and int(u.split()[0])==0 else 0;time.sleep(1)
  tag=f'{d}_{phase}_b{b}_{s}_004';env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');log=C.ART/'logs'/f'{tag}.log'
  with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_003.py'),d,gpu,phase,str(b),s,'004'],stdout=f,stderr=subprocess.STDOUT,cwd=C.ROOT,env=env)
  runs.append({'tag':tag,'phase':phase,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(runs[-1]),flush=True)
  if r.returncode:raise RuntimeError('Preserve failure, stop affected performance stage')
 C.write(C.ART/f'S1/{d}_{phase}_completion_004.json',{'time_bj':C.stamp(),'status':'PASS_COMPLETED','runs':[x for x in runs if x['phase']==phase]})
