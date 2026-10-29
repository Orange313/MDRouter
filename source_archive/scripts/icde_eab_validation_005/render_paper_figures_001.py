"""Rebuild paper figures from allowed archived CSV tables. No models, GPU, qrels or tuning."""
import argparse,pathlib,csv,os
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.root);out=pathlib.Path(a.output);out.mkdir(exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'mpl_cache');import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
def read(p):return list(csv.DictReader((root/p).open(encoding='utf8')))
def save(fig,name):
 for e in ['png','pdf']:fig.savefig(out/(name+'.'+e),dpi=190)
 plt.close(fig)
domains=['vidore_v3_hr','vidore_v3_finance_en','vidore_v3_industrial','Pharmaceuticals'];names=['HR','Finance-EN','Industrial','Pharma'];rows=read('reports/icde_eab_validation_005/A1/matched_risk_cost_master.csv');fig,ax=plt.subplots(2,4,figsize=(16,7))
for i,d in enumerate(domains):
 for r in rows:
  if r['domain']!=d:continue
  ax[0,i].scatter(float(r['actual_E2E_s'])*1000,float(r['mean_positive_regret']),label=r['method']);ax[1,i].scatter(float(r['actual_E2E_s'])*1000,float(r['mean_nDCG10']))
 ax[0,i].set(title=names[i],xlabel='Measured resident full-path ms',ylabel='Positive regret');ax[1,i].set(xlabel='Measured resident full-path ms',ylabel='Mean nDCG@10')
handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=8);fig.tight_layout(rect=(0,.12,1,1));save(fig,'quality_risk_cost')
fig,ax=plt.subplots(figsize=(10,5));labs=[]
for r in rows:
 if r['method'] not in ['Q_regret','QP_regret']:continue
 labs.append(names[domains.index(r['domain'])]+'/'+r['method']);v=float(r['increment_pp']);lo=float(r['paired95_low_pp']);hi=float(r['paired95_high_pp']);ax.errorbar(v,len(labs)-1,xerr=[[v-lo],[hi-v]],fmt='o',mfc='C0' if r['resident_AA']=='PASS' else 'white',color='C0',capsize=3)
ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(len(labs)),yticklabels=labs,xlabel='Increment vs ideal same-risk static/random (pp of FullV)',title='Dev exploratory; open markers: resident A/A insufficient');fig.tight_layout();save(fig,'matched_increment')
learning=read('reports/icde_final_gate_004/S3/matched_risk_learning_curves.csv');fig,ax=plt.subplots(2,2,figsize=(11,7))
for i,m in enumerate(['Q_regret','QP_regret']):
 for k in [0,1,5,10,20,50,100]:
  z=sorted([r for r in learning if r['method']==m and float(r['kappa'])==k],key=lambda r:int(r['Fit_n']));x=[int(r['Fit_n']) for r in z];ax[0,i].plot(x,[float(r['mean_Dev_r']) for r in z],'-o',markersize=3,label=f'kappa={k}');ax[1,i].plot(x,[float(r['matched_risk_increment_pp']) for r in z],'-o',markersize=3)
 ax[0,i].set(title=m,ylabel='Fixed Dev mean positive regret');ax[1,i].set(xlabel='Fit query count',ylabel='Matched static-random increment(pp)');ax[1,i].axhline(0,color='gray',lw=.7)
handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=7,loc='lower center',fontsize=8);fig.tight_layout(rect=(0,.07,1,1));save(fig,'learning_curves')
print('PASS_3_MAIN_FIGURES_FROM_VERSIONED_CSV_NO_NEW_EXPERIMENT')
