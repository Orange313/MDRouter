import common as C
import sys,subprocess,os,pathlib,json,time
phase=sys.argv[1];d='vidore_v3_hr';g=C.load(C.ART/f'S1/AA_gate_{d}_001.json');assert g['status']=='PASS';parent='S1' if phase=='stress' else 'S2';configs=[(0,s) for s in ['A1','B1','B2','A2']] if phase=='stress' else [(b,'R') for b in range(3)];rr=[]
C.write(C.ART/f'{parent}/hr_{phase}_pre_run_lock_008.json',{'time_bj':C.stamp(),'gate_SHA256':C.sha(C.ART/f'S1/AA_gate_{d}_001.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_005.py'),'protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'no_new_scientific_parameters':True})
for b,s in configs:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');tag=f'{d}_{phase}_b{b}_{s}_008';log=C.ART/'logs'/f'{tag}.log'
 with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'stream_worker_005.py'),d,'0',phase,str(b),s,'008'],cwd=C.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 rr.append({'tag':tag,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(rr[-1]),flush=True)
 if r.returncode:break
C.write(C.ART/f'{parent}/hr_{phase}_completion_008.json',{'runs':rr,'time_bj':C.stamp(),'all_complete':len(rr)==len(configs) and all(r['returncode']==0 for r in rr)})
