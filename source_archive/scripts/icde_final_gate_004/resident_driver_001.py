import common as C
import subprocess,pathlib,os,concurrent.futures,json
spec={'status':'LOCKED_BEFORE_NEW_RESIDENT_MEASUREMENTS','time_bj':C.stamp(),'purpose':'actual complete synchronous path cost, independent of stream superiority gate; no arrival schedule/queue/QPS/throughput claim','domains':C.DOM,'methods':C.PLANS+['Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret'],'queries':'all119oldDev +45PharmaDev; no newlabels','processes_per_domain':3,'repeats_per_query_method_process':3,'worker':'one persistent actual GPU thread,20FullV warmup then1permethod; all timed requests retained','order':'seeded randomized queries/methods per process/repeat, fixed seeds; unchanged frozen policies','GPU_profiles':'process0 secondary cohort allqueries/methods after primary; profile wall not primary; no forced kernel on CPU-only selected actions','AA_gate':'all three process-pair FullV mean ratios and paired family95%CI within [.90,1.10] before claiming resident superiority; if fail retain as noisy measurement','scope':'resident full-path service E2E inclQ/P/planning; no streaming/net business claim; setup separate','code_SHA256':{p.name:C.sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')}};C.write(C.ART/'S2/resident_protocol_001.json',spec)
assign={'vidore_v3_hr':1,'vidore_v3_finance_en':1,'vidore_v3_industrial':4,'Pharmaceuticals':4}
def lane(gpu):
 out=[]
 for d in C.DOM:
  if assign[d]!=gpu:continue
  for rep in range(3):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');log=C.ART/'logs'/f'resident_{d}_p{rep}_001.log'
   with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(pathlib.Path(__file__).parent/'resident_worker_001.py'),d,str(gpu),str(rep)],cwd=C.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   out.append({'domain':d,'process':rep,'returncode':r.returncode,'time_bj':C.stamp()});print(json.dumps(out[-1]),flush=True)
   if r.returncode:break
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rr=[r for z in ex.map(lane,[1,4]) for r in z]
C.write(C.ART/'S2/resident_driver_completion_001.json',{'runs':rr,'time_bj':C.stamp(),'all_successful':len(rr)==12 and all(x['returncode']==0 for x in rr)})
