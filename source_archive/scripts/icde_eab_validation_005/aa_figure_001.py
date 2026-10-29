import common as C,numpy as np
C.start('A4/AA_figure_001');rows=C.load(C.REPORT/'A2/aa_equivalence_v3_all_domains.csv');rows=[r for r in rows if r['ratio']];plt=C.plots();fig,ax=plt.subplots(figsize=(10,4.6));labels=[]
for i,r in enumerate(rows):
 labels.append(('HR' if r['domain']==C.DOM[0] else 'Pharma')+' / '+r['QPS']+' QPS / '+('service' if r['metric']=='mean_service_wall_s' else 'arrival-total'));v=float(r['ratio']);lo=float(r['CI_low']);hi=float(r['CI_high']);ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',mfc='C0' if r['status']=='PASS' else 'white',color='C0',capsize=3)
ax.axvspan(.9,1.1,color='gray',alpha=.13,label='Predeclared equivalence region');ax.axvline(1,color='black',lw=.6);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Paired process geometric mean ratio, Bonferroni98.75% CI',title='New005 A/A: neither domain passes all predeclared checks');ax.legend(fontsize=8);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(C.R/('new_AA_equivalence.'+ext),dpi=190)
plt.close(fig);C.write(C.A/'complete.json',{'status':'PASS_PLOT_OF_NEGATIVE_RELIABILITY_RESULT','time_bj':C.stamp(),'independent_process_pairs_per_domain':4,'new_measured_requests':2720})
