import common as C,pathlib,json,shutil,subprocess,numpy as np,collections,importlib.metadata as im
C.start('A4/draft_002');paper=C.R/'paper_source_001';paper.mkdir();(paper/'figures').mkdir();(paper/'tables').mkdir();old=C.ROOT/'reports/icde_final_gate_004';rows=C.load(C.REPORT/'A1/matched_risk_cost_master.csv');breaks=C.load(C.REPORT/'A1/probe_break_even.csv');comp=C.load(old/'S2/resident_analysis_002/full_path_cost_decomposition.csv');oracle=C.load(C.REPORT/'A1/oracle_gap_decomposition.csv');learning=C.load(C.PR/'E2/learning_curve_pharma.csv');limits=C.load(old/'S3/paired_learning_gain_matched_risk.csv')

for r in rows:
 for k in ['mean_nDCG10','mean_positive_regret','actual_E2E_s','increment_pp','paired95_low_pp','paired95_high_pp']:r[k]=float(r[k])
C.table(C.R/'paper_main_table.csv',rows);C.table(C.R/'paper_cost_components.csv',comp);C.table(C.R/'paper_oracle_table.csv',oracle);C.table(C.R/'paper_supervision_ablation.csv',[x for x in learning if x['source']=='Pharma_only' and x['Pharma_fit_n']=='81']);C.table(C.R/'paper_learning_changes.csv',limits)
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
tab=[r'\begin{table*}[t]',r'\centering\caption{Measured resident paths on repeatedly exposed Dev data. Increment is relative to an ideal same-risk static-random envelope, in percentage points of Full Visual cost. Intervals are conditional paired family+process bootstrap95\%; they are not independent-test guarantees. Only Finance resident A/A passed.}\label{tab:main}',r'\begin{tabular}{llrrrrl}\toprule Domain & Policy & nDCG@10 & Positive regret & E2E(ms) & Increment(pp) & Paired95\% interval\\\midrule']
for r in rows:
 if r['method'].startswith('RandomMix_QP'):continue
 tab.append(f'{C.NAMES[C.DOM.index(r["domain"])]} & {esc(r["method"])} & {r["mean_nDCG10"]:.4f} & {r["mean_positive_regret"]:.4f} & {1000*r["actual_E2E_s"]:.2f} & {r["increment_pp"]:.2f} & [{r["paired95_low_pp"]:.2f},{r["paired95_high_pp"]:.2f}]'+r'\\')
tab += [r'\bottomrule\end{tabular}\end{table*}'];C.prose(paper/'tables/main_table.tex','\n'.join(tab)+'\n')
plt=C.plots();fig,ax=plt.subplots(figsize=(10,5));labs=[]
for r in rows:
 if r['method'] not in ['Q_regret','QP_regret']:continue
 i=len(labs);labs.append(C.NAMES[C.DOM.index(r['domain'])]+'/'+r['method']);v=r['increment_pp'];ax.errorbar(v,i,xerr=[[v-r['paired95_low_pp']],[r['paired95_high_pp']-v]],fmt='o',mfc='C0' if r['resident_AA']=='PASS' else 'white',color='C0',capsize=3)
ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(len(labs)),yticklabels=labs,xlabel='Increment vs ideal same-risk static/random (pp of FullV)',title='Dev exploratory; open markers: resident A/A insufficient');fig.tight_layout();fig.savefig(paper/'figures/matched_increment.pdf');fig.savefig(paper/'figures/matched_increment.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,4,figsize=(15,4))
for di,d in enumerate(C.DOM):
 z=[x for x in comp if x['domain']==d];bottom=np.zeros(len(z));labels=[x['method'] for x in z]
 for field,label in [('Q_wall_s','Q feature'),('Probe_search_extra_wall_s','Extra probe'),('Pstats_extra_wall_s','Probe stats'),('prediction_decision_wall_s','Predict/allocate'),('selected_retrieval_path_wall_s','Retrieval')]:
  val=np.array([float(x[field])*1000 for x in z]);axs[di].bar(range(len(z)),val,bottom=bottom,label=label);bottom+=val
 axs[di].set(title=C.NAMES[di],ylabel='Sequential measured wall components(ms)',xticks=range(len(z)),xticklabels=labels);axs[di].tick_params(axis='x',labelrotation=75,labelsize=7)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',fontsize=8);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(paper/'figures/path_components.pdf');fig.savefig(paper/'figures/path_components.png',dpi=170);plt.close(fig)
for src,dst in [(C.REPORT/'A1/quality_risk_cost','quality_risk_cost'),(old/'S3/learning_curves_all_kappa','learning_curves'),(old/'S4/certification_power_limit','certification_power'),(old/'figures_v2_001/AA_equivalence','prior_aa_equivalence')]:
 for ext in ['png','pdf']:shutil.copyfile(str(src)+'.'+ext,paper/'figures'/(dst+'.'+ext))
for n in ['main.tex','references.bib']:shutil.copyfile(pathlib.Path(__file__).parent/'paper_assets'/n,paper/n)
for n in ['IEEEtran.cls','IEEEtran.bst']:shutil.copyfile(pathlib.Path(__file__).parent/'paper_assets'/n,paper/n)
env={p:im.version(p) for p in ['numpy','scipy','torch','transformers','colpali-engine','pandas','pyarrow']};env['python']=subprocess.check_output([str(C.ROOT/'.venv/bin/python'),'--version'],text=True).strip();env['base_git']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();C.write(C.R/'environment_versions.json',env);C.prose(C.R/'requirements_recorded.txt','\n'.join(k+'=='+v for k,v in env.items() if k not in ['python','base_git'])+'\n')
C.prose(C.R/'paper_tables_and_figures.md','# 正文表图与证据\n\n- Table1: frozen action catalog and dataset/exposure registry A0; oldDev119+PharmaDev45,122候选族。\n- Table2: paper_main_table.csv → tables/main_table.tex，四域全部静态/Q/QP/冻结随机，独立评估NOT_RUN。\n- Table3: paper_supervision_ablation.csv：既有同容量Q_regret/Q_hard/Q_soft/QP全部κ/3seed，成本为旧E2组合路径，不能混作004真实路径成本。不是原RetrievalRouter复现。\n- Fig1: quality_risk_cost，两套坐标分别nDCG与positive regret，绝不互换。\n- Fig2: matched_increment，paired族+进程区间，常驻AA不足空心。\n- Fig3: path_components，顺序wall子段；GPU/CPU计数不相加成wall。\n- Fig4: learning_curves，全κ、全部seed，不挑单工作点。\n- Fig5: certification_power，假设零损失IID容量参照。\n- Fig6: prior_aa_equivalence，标识上一轮；本轮新AA结果由A2最终状态覆盖讨论，不以旧PASS冒充新执行。\n\n所有重复请求仅计时重复，非新增独立质量样本。最终独立eval、全面公开artifact与稳定多策略Serving目前均未完成。\n')
C.write(C.A/'draft_prepared.json',{'time_bj':C.stamp(),'paper_path':str(paper),'new_model_training':0,'status':'ENGLISH_LATEX_DRAFT_WITH_ACTUAL_TABLES_FIGURES_PENDING_A2_FINAL_STATUS'})
print(json.dumps({'status':'PAPER_DRAFT_PREPARED','path':str(paper),'time_bj':C.stamp()}),flush=True)
