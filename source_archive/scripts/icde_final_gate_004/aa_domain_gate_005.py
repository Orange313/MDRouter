import common as C
from aa_selection_005 import primary
import numpy as np,sys,json
from scipy.stats import t
d=sys.argv[1];C.A=C.ART/'S1';C.R=C.REPORT/'S1';rr=[];selection=[]
for b in range(2):
 per={}
 for s in ['A1','B1','B2','A2']:
  p,extra,bad=primary(d,b,s);per[s]=C.load(p/'cell_rows.json');selection.append({'block':b,'slot':s,'primary':str(p),'unplanned_extra':[str(x) for x in extra],'invalid_retained':[str(x) for x in bad]})
 for qps in [2.,5.]:
  for aa,bb in [('A1','B1'),('A2','B2')]:
   a=next(x for x in per[aa] if x['QPS_offered']==qps);z=next(x for x in per[bb] if x['QPS_offered']==qps)
   for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:rr.append({'QPS':qps,'metric':k,'logratio':np.log(z[k]/a[k])})
out=[]
for qps in [2.,5.]:
 for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:
  z=np.array([x['logratio'] for x in rr if x['QPS']==qps and x['metric']==k]);mu=z.mean();hw=t.ppf(1-.0125/2,3)*z.std(ddof=1)/2;lo,hi=np.exp([mu-hw,mu+hw]);out.append({'QPS':qps,'metric':k,'ratio':np.exp(mu),'CI_low':lo,'CI_high':hi,'status':'PASS' if lo>=.9 and hi<=1.1 else 'INSUFFICIENT_EQUIVALENCE'})
result={'domain':d,'status':'PASS' if all(x['status']=='PASS' for x in out) else 'FAIL','statistical_status':'PASS_EQUIVALENCE' if all(x['status']=='PASS' for x in out) else 'INSUFFICIENT_EQUIVALENCE','gate_checks':out,'process_selection':selection,'time_bj':C.stamp(),'predeclared_protocol_SHA256':C.sha(C.A/'streaming_protocol_v2.json'),'unplanned_duplicate_policy':'not primary; all retained and separately disclosed, no outcome-based selection'}
fp=C.A/f'AA_gate_{d}_001.json'
if fp.exists():
 old=json.loads(fp.read_text());assert old['status']==result['status'];C.write(C.A/f'AA_selection_audit_{d}_005.json',result)
else:C.write(fp,result)
print(json.dumps(result),flush=True)
