import common as C
import os,sys,time,pathlib,subprocess,json
A=C.ART/'E2';index=C.ROOT/'cache/indexes/pharma_fixedcorpus_index_001/index_manifest.json';guard=A/'index_build_002/resource_guard.json'
while not(index.exists() and guard.exists()):
 if (A/'index_build_002/failure.json').exists():raise RuntimeError('Index failed; outcomes not started')
 time.sleep(5)
assert json.loads(guard.read_text())['valid'];assert (A/'public_projection_final_001/projection_manifest.json').exists()
while True:
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True);free=[r.split(',')[0].strip() for r in raw.strip().splitlines() if int(r.split(',')[1].split()[0])<128 and int(r.split(',')[2].split()[0])==0]
 if free:gpu='5' if '5' in free else free[0];break
 time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false');log=(A/'outcomes_driver_001.log').open('x');p=subprocess.run([sys.executable,str(C.ROOT/f'scripts/{C.RUN}/e2_outcomes_001.py'),gpu],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();print(json.dumps({'stage':'E2_OUTCOMES_DRIVER','GPU':gpu,'exit':p.returncode,'time_bj':C.stamp()}),flush=True)
if p.returncode:raise RuntimeError('Outcomes failed; preserved; no training')
with (A/'learning_driver_001.log').open('x') as log:
 p=subprocess.run([sys.executable,str(C.ROOT/f'scripts/{C.RUN}/e2_learning_001.py')],cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps({'stage':'E2_LEARNING_DRIVER','exit':p.returncode,'time_bj':C.stamp()}),flush=True)
if p.returncode:raise RuntimeError('Learning failed; preserved; no overwrite')
