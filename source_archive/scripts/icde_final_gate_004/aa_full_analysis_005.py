import common as C
from aa_selection_005 import primary
import numpy as np,json,itertools,collections
from scipy.stats import t
C.A=C.ART/'S1';C.R=C.REPORT/'S1';C.STAGE='S1';requests=[];pairs=[];selection=[];telemetry=[];cells=[]
for d in C.DOM:
 for b in range(2):
  per={}
  for s in ['A1','B1','B2','A2']:
   p,extra,bad=primary(d,b,s);per[s]=C.load(p/'cell_rows.json');cells+=per[s];g=C.load(p/'resource_guard.json');assert g['valid'];rr=C.load(p/'request_rows.json');requests.extend([{**r,'primary_scheduled':True,'attempt_dir':p.name} for r in rr]);selection.append({'domain':d,'block':b,'slot':s,'primary_attempt':p.name,'unplanned_extra':[x.name for x in extra],'invalid_attempts_retained':[x.name for x in bad]})
   tv=C.load(p/'telemetry.json');valid=[r for r in tv if 'error' not in r];ru=resource=None
   def avg(k):return float(np.mean([r[k] for r in valid if r.get(k) is not None])) if valid else None
   telemetry.append({'domain':d,'block':b,'slot':s,'attempt':p.name,'samples':len(valid),'GPU_SM_clock_MHz_mean':avg('GPU_SM_clock_MHz'),'GPU_util_pct_mean':avg('GPU_SM_util_pct'),'MemAvailable_kB_mean':avg('MemAvailable_kB'),'context_switch_nonvoluntary_first_last':valid[-1]['involuntary_context_switches']-valid[0]['involuntary_context_switches'] if valid else None,'host_load_mean':np.mean([r['loadavg'][0] for r in valid]) if valid else None,'service_ms_mean':1000*np.mean([r['service_wall_s'] for r in rr]),'queue_ms_mean':1000*np.mean([r['executor_queue_s'] for r in rr]),'dispatch_lateness_ms_mean':1000*np.mean([r['arrival_dispatch_lateness_s'] for r in rr]),'causality':'ASSOCIATION_ONLY_NO_SINGLE_CAUSE_IDENTIFIED'})
   for ex in extra:requests.extend([{**r,'primary_scheduled':False,'attempt_dir':ex.name,'status':'UNPLANNED_DUPLICATE_RETAINED_NOT_PRIMARY_GATE'} for r in C.load(ex/'request_rows.json')])
  for qps in [2.,5.]:
   for aa,bb in [('A1','B1'),('A2','B2')]:
    a=next(r for r in per[aa] if r['QPS_offered']==qps);z=next(r for r in per[bb] if r['QPS_offered']==qps)
    for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:pairs.append({'domain':d,'block':b,'A':aa,'B':bb,'QPS':qps,'metric':k,'A_mean':a[k],'B_mean':z[k],'logratio':np.log(z[k]/a[k])})
summary=[];gate={'time_bj':C.stamp(),'domains':{},'threshold':[.9,1.1],'CI_confidence':.9875,'paired_process_n':4,'scope':'approximate Student-t paired process logmeans; df3; Bonferroni four checks/domain; no population risk guarantee; four pair power limited'}
for d in C.DOM:
 for qps in [2.,5.]:
  for k in ['mean_service_wall_s','mean_arrival_to_completion_s']:
   z=np.array([r['logratio'] for r in pairs if r['domain']==d and r['QPS']==qps and r['metric']==k]);mu=z.mean();hw=t.ppf(1-.0125/2,3)*z.std(ddof=1)/2;lo,hi=np.exp([mu-hw,mu+hw]);summary.append({'domain':d,'QPS':qps,'metric':k,'paired_n':4,'B_A_ratio':np.exp(mu),'CI_low':lo,'CI_high':hi,'status':'PASS' if lo>=.9 and hi<=1.1 else 'INSUFFICIENT_EQUIVALENCE','operational_gate':'PASS' if lo>=.9 and hi<=1.1 else 'FAIL_GATE','CI_scope':gate['scope']})
 rr=[r for r in summary if r['domain']==d];passed=all(r['status']=='PASS' for r in rr);gate['domains'][d]={'status':'PASS' if passed else 'FAIL','checks':rr};fp=C.A/f'AA_gate_{d}_001.json'
 if fp.exists():assert json.loads(fp.read_text())['status']==gate['domains'][d]['status']
 else:C.write(fp,{'domain':d,**gate['domains'][d],'time_bj':C.stamp()})
C.table(C.R/'AA_independent_process_pairs.csv',pairs);C.table(C.R/'AA_equivalence_gate.csv',summary);C.table(C.R/'aa_per_request.csv',requests);C.table(C.R/'new_queue_service_cells.csv',cells);C.table(C.R/'process_selection_and_failures.csv',selection);C.table(C.R/'CPU_GPU_telemetry_summary.csv',telemetry);C.write(C.A/'AA_gate.json',gate)
plt=C.plots();fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,metric in zip(axes,['mean_service_wall_s','mean_arrival_to_completion_s']):
 for j,d in enumerate(C.DOM):
  for qi,qps in enumerate([2.,5.]):
   r=next(x for x in summary if x['domain']==d and x['QPS']==qps and x['metric']==metric);ax.errorbar(r['B_A_ratio'],j+(qi-.5)*.18,xerr=[[r['B_A_ratio']-r['CI_low']],[r['CI_high']-r['B_A_ratio']]],fmt='o',label=f'{qps:g} QPS' if j==0 else None)
 ax.axvspan(.9,1.1,color='green',alpha=.15);ax.axvline(1,color='gray',lw=.7);ax.set(title=metric.replace('mean_',''),xlabel='B/A paired process ratio, 98.75% CI',yticks=range(4),yticklabels=C.NAMES);ax.legend()
fig.tight_layout();fig.savefig(C.R/'AA_equivalence.png',dpi=180);fig.savefig(C.R/'AA_equivalence.pdf');plt.close(fig)
print(json.dumps({'status':'AA_GATE_COMPLETE','domains':{d:v['status'] for d,v in gate['domains'].items()},'primary_requests':sum(r['primary_scheduled'] for r in requests),'extra_requests':sum(not r['primary_scheduled'] for r in requests),'time_bj':C.stamp()}),flush=True)
