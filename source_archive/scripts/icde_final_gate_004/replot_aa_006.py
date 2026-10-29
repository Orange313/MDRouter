import common as C
import pathlib,sys,numpy as np
out=pathlib.Path(sys.argv[1]);out=out if out.is_absolute() else C.ROOT/out;out.mkdir(parents=True,exist_ok=False);C.A=out;C.R=out
rows=C.load(C.REPORT/'S1/AA_equivalence_gate.csv');plt=C.plots();fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,metric in zip(axes,['mean_service_wall_s','mean_arrival_to_completion_s']):
 for j,d in enumerate(C.DOM):
  for qi,qps in enumerate([2.,5.]):
   z=next(r for r in rows if r['domain']==d and float(r['QPS'])==qps and r['metric']==metric);v=float(z['B_A_ratio']);lo=float(z['CI_low']);hi=float(z['CI_high']);ax.errorbar(v,j+(qi-.5)*.18,xerr=[[v-lo],[hi-v]],color=['C0','C1'][qi],fmt='o',label=f'{qps:g} QPS' if j==0 else None)
 ax.axvspan(.9,1.1,color='green',alpha=.15);ax.axvline(1,color='gray',lw=.7);ax.set(title=metric.replace('mean_',''),xlabel='B/A paired process ratio, 98.75% CI',yticks=range(4),yticklabels=C.NAMES);ax.legend()
fig.tight_layout();fig.savefig(out/'AA_equivalence.png',dpi=180);fig.savefig(out/'AA_equivalence.pdf');plt.close(fig);C.write(out/'render_audit.json',{'time_bj':C.stamp(),'status':'PASS_STYLE_CORRECTION_ONLY','source_SHA256':C.sha(C.REPORT/'S1/AA_equivalence_gate.csv'),'reason':'original plot auto-cycled color by domain although legend keyedQPS; v2 fixes consistent QPS colors, no numeric/interval changes; old figure retained'})
