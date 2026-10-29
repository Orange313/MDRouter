"""One command redraws figures from immutable measured tables; never modifies old figures."""
import common as C
import pathlib,sys,os,argparse,csv,json,numpy as np
def run(out):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'mpl_cache');sys.path.insert(0,str(C.ROOT/'cache/report_dependencies/prereg_v2_round_001'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 def rows(p):return list(csv.DictReader((C.REPORT/p).open(encoding='utf8')))
 def save(fig,name):
  fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=220);fig.savefig(out/(name+'.pdf'));plt.close(fig)
 static=rows('E1/static_quality_e2e.csv');learning=rows('E2/learning_curve_pharma.csv');stream=rows('E4/streaming_window_sweep.csv');names=dict(zip(C.DOM,C.NAMES))
 fig,ax=plt.subplots(1,3,figsize=(15,4))
 for a,d in zip(ax,C.DOM):
  for r in static:
   if r['domain']!=d:continue
   x=float(r['mean_resident_E2E_s'])*1000;y=float(r['mean_r']);lo=float(r['r_CI_low']);hi=float(r['r_CI_high']);a.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=3,label=r['plan'])
  a.set(title=names[d],xlabel='Resident full-request mean E2E (ms)',ylabel='Mean positive nDCG10 regret')
 ax[0].legend(loc='upper right',fontsize=7)
 fig.suptitle('Measured static paths; Dev conditional pointwise family intervals; no certificate');save(fig,'figure1_static_risk_cost')
 fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(3);riskdrop=[];savings=[];lo=[];hi=[]
 for d in C.DOM:
  a=next(r for r in static if r['domain']==d and r['plan']=='BM25-R50');b=next(r for r in static if r['domain']==d and r['plan']=='BM25-R100');riskdrop.append(100*(1-float(b['mean_r'])/float(a['mean_r'])));savings.append(float(b['E2E_saving_pct']));lo.append(float(b['paired_E2E_saving_CI_low_pct']));hi.append(float(b['paired_E2E_saving_CI_high_pct']))
 ax[0].bar(x,riskdrop);ax[0].set(xticks=x,xticklabels=C.NAMES,ylabel='R100 vs R50 mean positive-regret reduction (%)');ax[1].bar(x,savings,yerr=[np.array(savings)-lo,np.array(hi)-savings],capsize=4);ax[1].axhline(10,color='grey',ls='--');ax[1].set(xticks=x,xticklabels=C.NAMES,ylabel='R100 vs FullV resident E2E saving (%)');fig.suptitle('Risk reduction differs from repairing every harmed query');save(fig,'figure2_candidate_repair_runtime')
 fig,ax=plt.subplots(1,3,figsize=(14,4));methods=['Q_regret','Q_hard','Q_soft','QP_regret']
 for method in methods:
  nn=[20,40,60,81];values=[[],[],[]];errors=[[],[],[]]
  for n in nn:
   z=[r for r in learning if r['source']=='Pharma_only' and r['method']==method and int(r['Pharma_fit_n'])==n and float(r['kappa'])==10]
   for j,key in enumerate(['mean_r','E2E_saving_pct','false_safe_rate_r_gt05']):
    v=np.array([float(r[key]) for r in z]);values[j].append(v.mean());errors[j].append(v.std())
  for j in range(3):ax[j].errorbar(nn,values[j],yerr=errors[j],fmt='-o',label=method,capsize=3)
 for a,label in zip(ax,['Fixed Dev mean positive regret','Measured-path E2E saving (%)','False-Safe fraction: chosen non-V and r>.05']):a.set(xlabel='Pharma Fit query budget',ylabel=label)
 ax[0].legend(fontsize=7);fig.suptitle('Fixed kappa=10 illustration; bars = 3 training-seed SD, not independent-data CI; all kappa in CSV');save(fig,'figure3_training_budget')
 fig,ax=plt.subplots(2,2,figsize=(12,9))
 for a,d in zip(ax.flat,C.DOM+['Pharmaceuticals']):
  z=[r for r in stream if r['domain']==d and int(r['window'])==1 and int(r['concurrency'])==1 and int(r['load_index'])==1]
  colors=dict(zip(['BM25','BM25-R50','BM25-R100','Full-Visual','Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret'],plt.get_cmap('tab10').colors))
  for r in z:a.scatter(float(r['mean_latency_s'])*1000,float(r['mean_r']),label=r['method'],color=colors[r['method']])
  a.legend(loc='upper right',fontsize=7)
  a.set(title=names.get(d,d),xlabel='Actual arrival-to-completion mean latency (ms)',ylabel='Realized mean positive regret')
 fig.suptitle('Fixed W1 / C1 / medium load; A/A reliability FAILED: descriptive only');save(fig,'figure4_stream_realized_risk_latency')
 fig,ax=plt.subplots(figsize=(7,4));nn=np.arange(25,1001)
 for m in [1,3,7]:ax.plot(nn,1-(.01/m)**(1/nn),label=f'{m} fixed non-reference hypotheses')
 ax.axhline(.05,color='grey',ls='--',label='epsilon=.05');ax.axvline(81,color='black',ls=':');ax.set(xlabel='Hypothetical independent all-zero-loss calibration n',ylabel='Exact all-zero upper bound',ylim=(0,.18),title='Ideal IID power only; actual Pharma independence not established');ax.legend();save(fig,'figure5_ideal_calibration_power')
 inputs={str(C.REPORT/p):C.sha(C.REPORT/p) for p in ['E1/static_quality_e2e.csv','E2/learning_curve_pharma.csv','E4/streaming_window_sweep.csv']};(out/'figure_input_hashes.json').write_text(json.dumps({'time_bj':C.stamp(),'code_sha256':C.sha(__file__),'inputs':inputs,'limits':'All Dev/exploratory or measured harness; no formal new K1; seed/time repeats are not independent query samples'},indent=2),encoding='utf8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
