import common as C,frontier_math as F
import numpy as np,json,collections,itertools
from scipy.optimize import linprog
C.start('A1');OLD=C.ROOT/'reports/icde_final_gate_004/S2/resident_analysis_002';raw=C.load(OLD/'all_unprofiled_requests.csv');prior=C.load(OLD/'static_vs_router_matched_risk.csv');cand=C.load('reports/icde_decisive_suite_002/R2/candidate_cover_vs_quality.csv');pcand=C.load(C.PREV/'E2/outcomes_001/outcomes.json');cand += [r for r in pcand if r['role']=='development'];lookup={(r['domain'],int(r['query_id']),r['plan']):r for r in cand};methods=C.PLANS+['Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret'];master=[];breaks=[];oracles=[];case=[];perquery=[];front=[];checks=[];B=2000
for di,d in enumerate(C.DOM):
 rr=[x for x in raw if x['domain']==d];ids=sorted({int(x['query_id']) for x in rr});N=len(ids);family=np.array([next(x['family'] for x in rr if int(x['query_id'])==q) for q in ids]);T=np.empty((N,3,8));Y=np.empty_like(T);U=np.empty_like(T);PATH=np.empty_like(T);OH=np.empty_like(T);CPU=np.empty_like(T);ACT={}
 for qi,q in enumerate(ids):
  for p in range(3):
   for mi,m in enumerate(methods):
    z=[x for x in rr if int(x['query_id'])==q and int(x['process'])==p and x['method']==m];assert len(z)==3
    mean=lambda k:np.mean([float(x[k]) for x in z]);T[qi,p,mi]=mean('resident_full_path_E2E_s');Y[qi,p,mi]=mean('r');U[qi,p,mi]=mean('nDCG10');CPU[qi,p,mi]=mean('CPU_thread_s');PATH[qi,p,mi]=mean('selected_path_wall_s');OH[qi,p,mi]=sum(mean(k) for k in ['Q_wall_s','Probe_search_extra_wall_s','Pstats_extra_wall_s','prediction_decision_wall_s']);ACT[qi,p,mi]=[x['action'] for x in z]
 tx=T.mean((0,1));yx=Y.mean((0,1));ux=U.mean((0,1));ixs=F.boots(family,B,C.SEED+di);rng=np.random.default_rng(C.SEED+di);ps=rng.integers(0,3,(B,3));BT=np.array([T[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)]);BY=np.array([Y[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)]);BP=np.array([PATH[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)]);BO=np.array([OH[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)])
 residentAA=next(x['resident_AA_status'] for x in prior if x['domain']==d)
 for mi,m in enumerate(methods):
  ev,w=F.envelope(yx[:4],tx[:4],yx[mi]);eb,_=F.envelope(BY[:,:4],BT[:,:4],BY[:,mi]);gain=100*(eb-BT[:,mi])/BT[:,0];lo,hi=np.quantile(gain,[.025,.975]);samecost,_=F.envelope(tx[:4],yx[:4],tx[mi]);old=next(x for x in prior if x['domain']==d and x['method']==m)
  for k,v in [('mean_r',yx[mi]),('actual_resident_full_path_mean_E2E_s',tx[mi]),('matched_risk_increment_pp',100*(ev[0]-tx[mi])/tx[0]),('paired95_increment_low_pp',lo),('paired95_increment_high_pp',hi)]:
   delta=abs(v-float(old[k]));checks.append(dict(domain=d,method=m,metric=k,prior=float(old[k]),recomputed=v,absolute_difference=delta,status='PASS' if delta<1e-10 else 'FAIL'));assert delta<1e-10
  staticok=[j for j in range(4) if yx[j]<=yx[mi]+1e-12];best=min(staticok,key=lambda j:tx[j]);master.append(dict(domain=d,method=m,Dev_n=N,candidate_families=len(set(family)),mean_nDCG10=ux[mi],mean_positive_regret=yx[mi],harm_rate=(Y[:,:,mi].mean(1)>1e-10).mean(),actual_E2E_s=tx[mi],CPU_thread_s=CPU[:,:,mi].mean(),same_risk_best_single_static=C.PLANS[best],same_risk_best_single_static_E2E_s=tx[best],same_risk_ideal_static_random_E2E_s=ev[0],ideal_mix_weights=w[0].tolist(),increment_pp=100*(ev[0]-tx[mi])/tx[0],paired95_low_pp=lo,paired95_high_pp=hi,matched_cost_ideal_static_random_r=samecost[0],actual_policy_minus_matchedcost_r=yx[mi]-samecost[0],resident_AA=residentAA,status='EXPLORATORY_NO_NEW_CERTIFICATE',ideal_envelope='fractional static random mixture, allocation overhead=0; NOT actual stream latency'))
  for space,vals in [('positive_regret_cost',yx),('negative_mean_nDCG_cost',-ux)]:
   dom=[methods[j] for j in range(4) if vals[j]<=vals[mi]+1e-12 and tx[j]<=tx[mi]+1e-12 and (vals[j]<vals[mi]-1e-12 or tx[j]<tx[mi]-1e-12)]
   front.append(dict(domain=d,method=m,space=space,mean_r=yx[mi],mean_nDCG=ux[mi],E2E_s=tx[mi],static_dominated_by=dom,status='POINT_DESCRIPTIVE'))
  if m in ['Q_regret','QP_regret']:
   budget=ev[0]-PATH[:,:,mi].mean();paid=tx[mi]-PATH[:,:,mi].mean();bootbudget=eb-BP[:,mi];bootpaid=BT[:,mi]-BP[:,mi];low,high=np.quantile(bootbudget-bootpaid,[.025,.975]);breaks.append(dict(domain=d,method=m,mean_executed_retrieval_path_s=PATH[:,:,mi].mean(),maximum_affordable_nonpath_wall_s=budget,actual_nonpath_wall_s=paid,measured_Q_probe_stats_predict_wall_s=OH[:,:,mi].mean(),remaining_break_even_margin_s=budget-paid,paired95_margin_low_s=low,paired95_margin_high_s=high,overhead_reduction_needed_s=max(0,paid-budget),probe_reuse='QP BM path reuses search once; FullV pays extra probe',interpretation='Break-even relative actual executed retrieval path; includes path context, not additive CPU+GPU; negative allowance cannot be fixed by free routing',resident_AA=residentAA,status='EXPLORATORY'))
   for catalog in [[0,2,3],[0,1,2,3]]:
    tt=T[:,:,:4].mean(1)[:,catalog];yy=Y[:,:,:4].mean(1)[:,catalog];lp=linprog(tt.ravel(),A_ub=[yy.ravel()],b_ub=[yx[mi]*N],A_eq=np.kron(np.eye(N),np.ones((1,len(catalog)))),b_eq=np.ones(N),bounds=(0,1),method='highs');assert lp.success
    allocation=lp.x.reshape(N,len(catalog));ow=(allocation*tt).sum(1);goldcost=lp.fun/N;selectionloss=tx[mi]-goldcost;idealstatic=F.envelope(yx[catalog],tx[catalog],yx[mi])[0][0];ci=np.quantile([100*(T[ix,:,mi].mean()-ow[ix].mean())/T[ix,:,0].mean() for ix in ixs],[.025,.975]);oracles.append(dict(domain=d,method=m,catalog=[C.PLANS[j] for j in catalog],mean_r_budget=yx[mi],reference_cost_s=tx[0],static_random_cost_s=idealstatic,offline_gold_oracle_cost_s=goldcost,learned_measured_cost_s=tx[mi],information_opportunity_pp=100*(idealstatic-goldcost)/tx[0],oracle_to_policy_loss_pp=100*selectionloss/tx[0],fixed_Dev_oracle_conditional95_low_pp=ci[0],fixed_Dev_oracle_conditional95_high_pp=ci[1],oracle_r=(allocation*yy).sum()/N,status='OFFLINE_ORACLE_NOT_DEPLOYABLE',CI_scope='candidate-family conditional contrast to allocation fitted on same Dev; not independent generalization nor re-certified risk'))
  for qi,q in enumerate(ids):
   z=[x for x in rr if int(x['query_id'])==q and x['method']==m];r=Y[qi,:,mi].mean();actions=collections.Counter(x['action'] for x in z);miss=0.;known=True
   for act,count in actions.items():
    c=lookup.get((d,q,act));has=c is not None and 'reference_positive_top10_missing' in c
    if has:miss += count*int(c['reference_positive_top10_missing'])/len(z)
    else:known=False
   row=dict(domain=d,query_id=q,family=family[qi],method=m,nDCG10=U[qi,:,mi].mean(),reference_nDCG10=U[qi,:,0].mean(),signed_gap=U[qi,:,0].mean()-U[qi,:,mi].mean(),positive_regret=r,harm=r>1e-10,false_safe=r>.05 and m in ['Q_regret','QP_regret'],actual_E2E_s=T[qi,:,mi].mean(),CPU_thread_s=CPU[qi,:,mi].mean(),executed_path_s=PATH[qi,:,mi].mean(),measured_preprocessing_s=OH[qi,:,mi].mean(),actions=dict(actions),mean_omitted_positive_reference_pages=miss if known else None,regret_contribution_if_candidate_miss=r if known and miss>0 else (0 if known else None),repeated_timing_n=9,evaluation='DEV_EXPLORATORY')
   perquery.append(row)
   if m in ['Q_regret','QP_regret'] and r>.05:case.append(row)
C.table(C.R/'matched_risk_cost_master.csv',master);C.table(C.R/'static_frontiers.csv',front);C.table(C.R/'oracle_gap_decomposition.csv',oracles);C.table(C.R/'probe_break_even.csv',breaks);C.table(C.R/'per_query_routing_diagnostics.csv',perquery);C.table(C.R/'false_safe_cases.csv',case);C.table(C.R/'historical_numeric_verification.csv',checks)
summ=[]
for d in C.DOM:
 for m in ['Q_regret','QP_regret']:
  z=[x for x in perquery if x['domain']==d and x['method']==m];total=sum(x['positive_regret'] for x in z);summ.append(dict(domain=d,method=m,false_safe_n=sum(x['false_safe'] for x in z),false_safe_loss_fraction=sum(x['positive_regret'] for x in z if x['false_safe'])/total if total else None,candidate_miss_loss_fraction=sum(x['positive_regret'] for x in z if (x['mean_omitted_positive_reference_pages'] or 0)>0)/total if total else None,status='DESCRIPTIVE_ASSOCIATION_NOT_CAUSAL'))
C.table(C.R/'loss_attribution.csv',summ)
C.prose(C.R/'error_case_analysis.md','# EXPLORATORY｜错误案例与开销归因\n\n所有False-Safe条目完整保留于CSV；定义为所选非参考计划r>.05，不是平均风险合同违约。候选遗漏与损失只是相关归因：BM25无视觉重排时即使候选包含相关页也可能排序失败。\n\n'+ '\n'.join(f'- {x["domain"]}/{x["method"]}: {x["false_safe_n"]} False-Safe；其遗憾占比{x["false_safe_loss_fraction"]}；伴随参考正相关页遗漏的遗憾占比{x["candidate_miss_loss_fraction"]}。' for x in summ)+'\n\n具体前12个高损失例：\n'+ '\n'.join(f'- {x["domain"]} q{x["query_id"]} {x["method"]}: actions={x["actions"]}, r={x["positive_regret"]:.5f}, E2E={1000*x["actual_E2E_s"]:.2f}ms, omitted-positive={x["mean_omitted_positive_reference_pages"]}' for x in sorted(case,key=lambda z:-z['positive_regret'])[:12])+'\n\n查询文本/页面内容可在合法Dev输入追溯；不将错误原因强行归为单一机制。\n')
plt=C.plots();fig,ax=plt.subplots(2,4,figsize=(16,7))
for di,d in enumerate(C.DOM):
 for x in master:
  if x['domain']!=d:continue
  ax[0,di].scatter(x['actual_E2E_s']*1000,x['mean_positive_regret'],label=x['method']);ax[1,di].scatter(x['actual_E2E_s']*1000,x['mean_nDCG10'])
 ax[0,di].set(title=C.NAMES[di],xlabel='Measured resident full-path ms',ylabel='Positive regret');ax[1,di].set(xlabel='Measured resident full-path ms',ylabel='Mean nDCG@10')
handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=8);fig.tight_layout(rect=(0,.12,1,1));fig.savefig(C.R/'quality_risk_cost.png',dpi=180);fig.savefig(C.R/'quality_risk_cost.pdf');plt.close(fig)
facts=f'PASS {len(raw)}次历史完整路径计时重算，{len(checks)}项主表/paired区间与004差<1e-10；164条Dev/122候选族，非独立测试。四域Q/QP均未证明稳定优于同风险静态随机包络；完整Oracle、错选损失、Probe break-even及False-Safe已输出。常驻AA仅Finance通过。'
C.prose(C.R/'A1_decision.md','# INSUFFICIENT｜动态路由方法收益未被证明\n\n'+facts+'\n\n同风险包络是零分配开销的理想静态随机混合，不是实测流式混合；冻结RandomMix的真实完整路径另列。所有区间为2000次配对候选query-family+进程bootstrap，逐点描述性，不支持未来查询PAC或独立测试显著性。Oracle使用同Dev gold，仅是信息上界。三域常驻A/A不足不能靠重算提升证据等级。\n\n停止复杂路由方法投资，保留计划设计、信息限制、统计功效与测量可靠性作为EAB分析。\n')
C.finish('PASS_REANALYSIS_METHOD_BENEFIT_INSUFFICIENT',facts,'动态路由方法收益未被证明；停止复杂模型；进入可信Serving门槛及独立评价准备。')
