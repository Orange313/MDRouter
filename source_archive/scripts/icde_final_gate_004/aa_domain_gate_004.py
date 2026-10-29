import common as C
import numpy as np,sys,json
from scipy.stats import t
d=sys.argv[1];C.A=C.ART/'S1';C.R=C.REPORT/'S1';rr=[]
for b in range(2):
 per={}
 for s in ['A1','B1','B2','A2']:
  attempt='002' if d=='vidore_v3_finance_en' or (d in ['vidore_v3_hr','Pharmaceuticals'] and (b!=0 or s not in ['A1','B1'])) else '001';p=C.A/f'workers/{d}_light_b{b}_{s}_{attempt}';per[s]=C.load(p/'cell_rows.json');assert C.load(p/'resource_guard.json')['valid']
 for qps in [2.,5.]:
  for aa,bb in [('A1','B1'),('A2','B2')]:
   a=next(x for x in per[aa] if x['QPS_offered']==qps);z=next(x for x in per[bb] if x['QPS_offered']==qps)
   for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:rr.append({'QPS':qps,'metric':k,'logratio':np.log(z[k]/a[k])})
out=[]
for qps in [2.,5.]:
 for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:
  z=np.array([x['logratio'] for x in rr if x['QPS']==qps and x['metric']==k]);mu=z.mean();hw=t.ppf(1-.0125/2,3)*z.std(ddof=1)/2;lo,hi=np.exp([mu-hw,mu+hw]);out.append({'QPS':qps,'metric':k,'ratio':np.exp(mu),'CI_low':lo,'CI_high':hi,'status':'PASS' if lo>=.9 and hi<=1.1 else 'FAIL'})
result={'domain':d,'status':'PASS' if all(x['status']=='PASS' for x in out) else 'FAIL','gate_checks':out,'time_bj':C.stamp(),'predeclared_protocol_SHA256':C.sha(C.A/'streaming_protocol_v2.json'),'note':'single scheduled per-domain gate; no repeated data collection or protocol tuning'}
C.write(C.A/f'AA_gate_{d}_001.json',result);print(json.dumps(result),flush=True)
