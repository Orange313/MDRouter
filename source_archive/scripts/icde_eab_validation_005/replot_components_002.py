import common as C,numpy as np
C.start('A4/figure_style_002');comp=C.load('reports/icde_final_gate_004/S2/resident_analysis_002/full_path_cost_decomposition.csv');plt=C.plots();fig,axs=plt.subplots(1,4,figsize=(15,4.2))
for di,d in enumerate(C.DOM):
 z=[x for x in comp if x['domain']==d];bottom=np.zeros(len(z))
 for field,label in [('Q_wall_s','Q features'),('Probe_search_extra_wall_s','Extra probe'),('Pstats_extra_wall_s','Probe stats'),('prediction_decision_wall_s','Predict/allocate'),('selected_retrieval_path_wall_s','Retrieval')]:
  val=np.array([float(x[field])*1000 for x in z]);axs[di].bar(range(len(z)),val,bottom=bottom,label=label);bottom+=val
 axs[di].set(title=C.NAMES[di],ylabel='Wall time(ms)',xticks=range(len(z)),xticklabels=[x['method'] for x in z]);axs[di].tick_params(axis='x',labelrotation=75,labelsize=7)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',fontsize=8);fig.tight_layout(rect=(0,.11,1,1));fig.savefig(C.R/'path_components.pdf');fig.savefig(C.R/'path_components.png',dpi=180);plt.close(fig);C.write(C.A/'style_change.json',{'status':'STYLE_ONLY_NO_NUMERICAL_CHANGE','reason':'shorten y-axis labels to avoid clipping','time_bj':C.stamp()})
