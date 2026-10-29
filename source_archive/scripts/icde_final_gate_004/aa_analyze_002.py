import common as C
import numpy as np,json,collections
from scipy.stats import t
C.A=C.ART/'S1';C.R=C.REPORT/'S1';C.STAGE='S1';sys=__import__('sys');sys.addaudithook(C.protected)
rows=[];requests=[];checks=[]
for d in C.DOM:
 for block in range(2):
  per={}
  for s in ['A1','B1','B2','A2']:
   attempt='002' if d=='vidore_v3_finance_en' else '001'
   p=C.A/f'workers/{d}_light_b{block}_{s}_{attempt}';per[s]=C.load(p/'cell_rows.json');r=C.load(p/'request_rows.json');requests+=r;g=C.load(p/'resource_guard.json');assert g['valid'];assert len(r)==4*({C.DOM[0]:40,C.DOM[1]:39,C.DOM[2]:40,'Pharmaceuticals':45}[d])
  for li in [0,1]:
   for aa,bb in [('A1','B1'),('A2','B2')]:
    a=next(x for x in per[aa] if x['load_index']==li);b=next(x for x in per[bb] if x['load_index']==li)
    for metric in ['mean_service_wall_s','mean_arrival_to_completion_s']:
     rows.append({'domain':d,'block':block,'A_slot':aa,'B_slot':bb,'QPS':a['QPS_offered'],'metric':metric,'A':a[metric],'B':b[metric],'log_ratio_B_A':np.log(b[metric]/a[metric]),'A_queue_s':a['mean_queue_s'],'B_queue_s':b['mean_queue_s'],'A_late_s':a['mean_arrival_dispatch_lateness_s'],'B_late_s':b['mean_arrival_dispatch_lateness_s'],'A_throughput':a['completed_QPS'],'B_throughput':b['completed_QPS']})
summary=[];gate={'time_bj':C.stamp(),'threshold':[.9,1.1],'within_domain_simultaneous_CI_confidence':.9875,'independent_pairs':4,'assumption':'approx normal independent paired process log means; only four pairs, conditional hardware/workload, no query IID claim','domains':{}}
for d in C.DOM:
 for qps in [2.,5.]:
  for metric in ['mean_service_wall_s','mean_arrival_to_completion_s']:
   rr=[x for x in rows if x['domain']==d and x['QPS']==qps and x['metric']==metric];z=np.array([r['log_ratio_B_A'] for r in rr]);mu=z.mean();width=t.ppf(1-.0125/2,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z));lo,hi=np.exp([mu-width,mu+width]);ratio=np.exp(mu);passed=.9<=lo and hi<=1.1 and .9<=ratio<=1.1
   summary.append({'domain':d,'QPS':qps,'metric':metric,'paired_n':len(z),'geometric_B_A_ratio':ratio,'simultaneous98_75_CI_low':lo,'simultaneous98_75_CI_high':hi,'margin_low':.9,'margin_high':1.1,'status':'PASS' if passed else 'FAIL_OR_INSUFFICIENT_EQUIVALENCE','limitation':gate['assumption']})
 rr=[x for x in summary if x['domain']==d];gate['domains'][d]={'status':'PASS' if all(x['status']=='PASS' for x in rr) else 'FAIL','failed_checks':[{'QPS':x['QPS'],'metric':x['metric'],'ratio':x['geometric_B_A_ratio'],'CI':[x['simultaneous98_75_CI_low'],x['simultaneous98_75_CI_high']]} for x in rr if x['status']!='PASS']}
C.table(C.R/'AA_independent_pairs.csv',rows);C.table(C.R/'AA_equivalence_gate.csv',summary);C.table(C.R/'aa_per_request.csv',requests);C.write(C.A/'AA_gate.json',gate)
plt=C.plots();fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,metric in zip(axes,['mean_service_wall_s','mean_arrival_to_completion_s']):
 for j,d in enumerate(C.DOM):
  for qi,qps in enumerate([2.,5.]):
   r=next(x for x in summary if x['domain']==d and x['QPS']==qps and x['metric']==metric);yy=j+(qi-.5)*.18;ax.errorbar(r['geometric_B_A_ratio'],yy,xerr=[[max(0,r['geometric_B_A_ratio']-r['simultaneous98_75_CI_low'])],[max(0,r['simultaneous98_75_CI_high']-r['geometric_B_A_ratio'])]],fmt='o',label=f'{qps:g} QPS' if j==0 else None)
 ax.axvspan(.9,1.1,alpha=.15,color='green');ax.axvline(1,color='gray',lw=.8);ax.set(title=metric.replace('mean_',''),xlabel='B/A paired ratio, 98.75% CI',yticks=range(4),yticklabels=C.NAMES);ax.legend()
fig.tight_layout();fig.savefig(C.R/'AA_equivalence.png',dpi=180);fig.savefig(C.R/'AA_equivalence.pdf');plt.close(fig)
print(json.dumps({'status':'AA_ANALYZED','gate':gate,'time_bj':C.stamp()},ensure_ascii=False),flush=True)

