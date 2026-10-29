import common as C
import learning_base_001 as L
import numpy as np,json,collections,time,sys,os
from scipy.optimize import linprog
C.A=C.ART/'E2';C.R=C.REPORT/'E2';C.STAGE='E2';plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];spec=C.load(C.A/'preanalysis.json');features=C.load(C.A/'outcomes_001/features.json','ALLOWED_FITDEV_FEATURES');out=C.load(C.A/'outcomes_001/outcomes.json','AUTHORIZED_FITDEV_OUTCOMES');qs=C.load(C.A/'allowed_fitdev_queries.json');fmap={int(r['query_id']):r for r in features};ol={(int(r['query_id']),r['plan']):r for r in out};ids=np.array([int(q['query_id']) for q in qs]);roles=np.array([q['role'] for q in qs]);g=np.array([q['family_id'] for q in qs]);assert len(ids)==126
Q=np.array([fmap[i]['Q'] for i in ids]);P=np.array([fmap[i]['P'] for i in ids]);Y=np.array([[ol[i,p]['r'] for p in plans] for i in ids]);T=np.array([[ol[i,p]['mean_E2E_s'] for p in plans] for i in ids]);G=np.array([[ol[i,p]['GPU_activity_s'] for p in plans] for i in ids]);CPU=np.array([[ol[i,p]['mean_CPU_s'] for p in plans] for i in ids]);qwall=np.array([fmap[i]['Q_wall_s'] for i in ids]);pwall=np.array([fmap[i]['Probe_wall_s'] for i in ids]);qcpu=np.array([fmap[i]['Q_CPU_s'] for i in ids]);pcpu=np.array([fmap[i]['Probe_CPU_s'] for i in ids]);train=np.flatnonzero(roles=='fit');val=np.flatnonzero(roles=='development');assert len(train)==81 and len(val)==45 and not(set(g[train])&set(g[val]));targetC=np.repeat((T[train].mean(0)/T[train,0].mean())[None],len(ids),axis=0)
old=[];e1=C.load('reports/icde_evidence_closure_003/E1/per_query_quality_cost.csv');e1lut={(r['domain'],int(r['query_id']),r['plan']):r for r in e1}
for d in C.DOM:
 a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz','ALLOWED_OLD_DEV_SOURCE_TRAINING');yy=np.array([[float(e1lut[d,int(i),p]['r']) for p in plans] for i in a['qids']]);tt=np.array([[float(e1lut[d,int(i),p]['mean_E2E_s']) for p in plans] for i in a['qids']]);old.append({'Q':a['Q'],'P':a['P'],'Y':yy,'T':tt,'C':np.repeat((tt.mean(0)/tt[:,0].mean())[None],len(tt),axis=0),'g':np.array([d+':'+x for x in a['family_primary']])})
old={k:np.concatenate([a[k] for a in old]) for k in old[0]};methods=['Q_regret','Q_hard','Q_soft','QP_regret'];summ=[];predrows=[];tunes=[];checks=[];models=[];B=2000;rng=np.random.default_rng(C.SEED);vg=g[val];u=np.unique(vg);boot=[np.concatenate([np.flatnonzero(vg==h) for h in rng.choice(u,len(u),replace=True)]) for _ in range(B)];iv=np.arange(len(val));staticrisk=Y[val].mean(0);statictime=T[val].mean(0)
def take(n,seed):
 order=sorted(set(g[train]),key=lambda x:__import__('hashlib').sha256((str(seed)+x).encode()).hexdigest());chosen=[]
 for h in order:
  ix=train[g[train]==h]
  if len(chosen)+len(ix)<=n:chosen.extend(ix.tolist())
 assert len(chosen)==n,(n,len(chosen));return np.array(chosen)
def staticmix(risk):
 z=linprog(statictime,A_ub=[staticrisk],b_ub=[risk+1e-12],A_eq=[np.ones(4)],b_eq=[1],bounds=(0,None),method='highs');assert z.success;return z.fun,z.x
def evaluate(source,n,seed,method,kap,Xtr,ytr,ctr,gtr):
 target='regret' if method.endswith('regret') else method.split('_')[-1];Xval=np.c_[Q[val],P[val]] if method.startswith('QP') else Q[val];l=L.tune(Xtr,ytr,ctr,gtr,target,kap,seed);m=L.fit(Xtr,L.targets(ytr,ctr,target,kap),l)
 t0=time.perf_counter();pred=L.predict(m,Xval);dec=L.actions(pred,targetC[val],target,kap);planning=(time.perf_counter()-t0)/len(val);r=Y[val][iv,dec];t=T[val][iv,dec]+qwall[val]+planning
 # P is paid once; if BM plan selected its own probe is reused. FullV pays extra probe.
 if method.startswith('QP'):
  # Reuse the measured BM probe in all BM-selected paths; FullV pays the entire probe.
  t=t+np.array([fmap[i]['Pstats_wall_s'] for i in ids[val]])+np.where(dec==0,np.array([fmap[i]['BM_search_wall_s'] for i in ids[val]]),0)
  scope='Q+P statistics+planning+measured path; BM search reused for BM arms; FullV pays extra BM search; actual stream verified in E4'
 else:scope='Q extraction+measured vectorized planning+full executed path'
 rv=float(r.mean());tv=float(t.mean());ref=T[val,0].mean();ci=np.quantile([100*(1-t[ix].mean()/T[val,0][ix].mean()) for ix in boot],[.025,.975]);rci=np.quantile([r[ix].mean() for ix in boot],[.025,.975]);mix,mixp=staticmix(rv)
 false=(dec!=0)&(r>.05);mse=np.mean((pred-Y[val])**2) if target=='regret' else None;den=np.sum((Y[val]-Y[val].mean(0))**2);r2=1-np.sum((pred-Y[val])**2)/den if target=='regret' and den>0 else None
 row={'source':source,'Pharma_fit_n':n,'seed':seed,'method':method,'kappa':kap,'L2':l,'training_queries':len(ytr),'training_candidate_families':len(set(gtr)),'Dev_queries':45,'Dev_candidate_families':len(u),'mean_r':rv,'r_CI_low':rci[0],'r_CI_high':rci[1],'harm_rate':np.mean(r>1e-10),'false_safe_rate_r_gt05':false.mean(),'mean_E2E_s':tv,'E2E_saving_pct':100*(1-tv/ref),'paired95_saving_low':ci[0],'paired95_saving_high':ci[1],'mean_GPU_activity_s':G[val][iv,dec].mean(),'native_risk_MSE':mse,'native_R2':r2,'matched_risk_best_static_random_mean_E2E_s':mix,'matched_risk_increment_E2E_saving_pct':100*(mix-tv)/ref,'mean_planning_s':planning,'allocation':np.bincount(dec,minlength=4).tolist(),'cost_scope':scope,'status':'FIXED_DEV_EXPLORATORY_UNCERTIFIED','interval':'paired fixed-Dev family bootstrap, conditional model; no IID/simultaneous claim'};summ.append(row)
 for i,qid in enumerate(ids[val]):predrows.append({'source':source,'Pharma_fit_n':n,'seed':seed,'method':method,'kappa':kap,'query_id':int(qid),'family':vg[i],'action':plans[dec[i]],'r':r[i],'E2E_s':t[i],'GPU_activity_s':G[val][i,dec[i]],'false_safe_r_gt05':bool(false[i]),'prediction':pred[i].tolist()})
 if source=='Pharma_only' and n==81:
  models.append({'source':source,'seed':seed,'method':method,'kappa':kap,'L2':l,'model':[x.tolist() for x in m],'cost_vector':targetC[0].tolist(),'actions':plans,'Dev_mean_r':rv,'Dev_mean_E2E_s':tv,'status':'DEV_SELECTED_CANDIDATE_NOT_CERTIFIED'})
 if source=='Pharma_only' and n==81 and seed==C.SEED and method=='Q_regret' and kap==10:
  # Held-out labels are not an argument to any fitting/tuning call; mutating them cannot alter predictions.
  altered=Y.copy();altered[val]=17.;assert np.array_equal(altered[train],Y[train]);m2=L.fit(Xtr,L.targets(ytr.copy(),ctr,target,kap),l);assert np.array_equal(pred,L.predict(m2,Xval));checks.append({'scope':'fixed Dev45 label mutation; training-only inputs unchanged','status':'PASS','Fit_Dev_families_disjoint':True})
 return row
for seed in spec['seeds']:
 for n in spec['n']:
  tr=take(n,seed)
  for source in ['Pharma_only','old119_plus_Pharma']:
   for method in methods:
    x=np.c_[Q[tr],P[tr]] if method.startswith('QP') else Q[tr];yy=Y[tr];cc=targetC[tr];gg=g[tr]
    if source!='Pharma_only':x=np.r_[np.c_[old['Q'],old['P']] if method.startswith('QP') else old['Q'],x];yy=np.r_[old['Y'],yy];cc=np.r_[old['C'],cc];gg=np.r_[old['g'],gg]
    for kap in spec['kappa']:evaluate(source,n,seed,method,kap,x,yy,cc,gg)
  print(json.dumps({'stage':'E2_LEARNING','seed':seed,'Pharma_fit_n':n,'completed_summary_rows':len(summ),'time_bj':C.stamp()}),flush=True)
 for method in methods:
  x=np.c_[old['Q'],old['P']] if method.startswith('QP') else old['Q']
  for kap in spec['kappa']:evaluate('old119_zero_shot',0,seed,method,kap,x,old['Y'],old['C'],old['g'])
C.table(C.R/'learning_curve_pharma.csv',summ);C.table(C.R/'crossdomain_transfer.csv',[r for r in summ if r['source']!='Pharma_only']);C.table(C.R/'supervision_ablation.csv',[r for r in summ if r['source']=='Pharma_only' and r['Pharma_fit_n']==81]);C.table(C.R/'per_plan_pharma_dev.csv',[r for r in out if r['role']=='development']);C.table(C.R/'per_query_policy.csv',predrows);C.write(C.A/'candidate_models.json',models);C.write(C.A/'heldout_mutation_checks.json',checks);C.write(C.A/'dataset_registry.json',{'query_ids':ids.tolist(),'roles':roles.tolist(),'families':g.tolist(),'actions':plans,'source_domains':C.DOM,'effective_IID_n':'UNKNOWN','training_source_oldDev_n':119,'target_fit_n':81,'target_Dev_n':45,'Cal_Eval_read_count':0})
gains=[]
for method in methods:
 for kap in spec['kappa']:
  base=[r for r in summ if r['source']=='Pharma_only' and r['Pharma_fit_n']==20 and r['method']==method and r['kappa']==kap];big=[r for r in summ if r['source']=='Pharma_only' and r['Pharma_fit_n']==81 and r['method']==method and r['kappa']==kap]
  gains.append({'method':method,'kappa':kap,'n20_mean_r':np.mean([r['mean_r'] for r in base]),'n81_mean_r':np.mean([r['mean_r'] for r in big]),'mean_r_change_81_minus20':np.mean([r['mean_r'] for r in big])-np.mean([r['mean_r'] for r in base]),'E2E_saving_change_81_minus20_pp':np.mean([r['E2E_saving_pct'] for r in big])-np.mean([r['E2E_saving_pct'] for r in base]),'matched_risk_increment_n20_pp':np.mean([r['matched_risk_increment_E2E_saving_pct'] for r in base]),'matched_risk_increment_n81_pp':np.mean([r['matched_risk_increment_E2E_saving_pct'] for r in big]),'native_R2_n20':np.mean([r['native_R2'] for r in base]) if method.endswith('regret') else None,'native_R2_n81':np.mean([r['native_R2'] for r in big]) if method.endswith('regret') else None,'status':'EXPLORATORY_FIXED_DEV_3_TRAINING_SEEDS_NOT_3_NEW_DATASETS'})
C.table(C.R/'training_gain_20_to81.csv',gains)
np.savez(C.A/'pharma_dataset.npz',qids=ids,roles=roles,family=g,Q=Q,P=P,Y=Y,T=T,GPU=G,CPU=CPU,qwall=qwall,pwall=pwall)
# Selection is explicitly Dev selection, never an independent evaluation or zero-shot score.
eligible=[r for r in summ if r['source']=='Pharma_only' and r['Pharma_fit_n']==81 and r['seed']==C.SEED and r['method'] in ['Q_regret','QP_regret'] and r['mean_r']<=.05]
selected=[]
for method in ['Q_regret','QP_regret']:
 choices=[r for r in eligible if r['method']==method]
 if choices:
  chosen=min(choices,key=lambda r:(r['mean_E2E_s'],-r['kappa']));selected.append(next(m for m in models if m['method']==method and m['kappa']==chosen['kappa'] and m['seed']==C.SEED))
C.write(C.A/'serving_candidate_models.json',selected)
sys.path.insert(0,str(C.ROOT/'cache/report_dependencies/prereg_v2_round_001'));os.environ['MPLCONFIGDIR']=str(C.A/'matplotlib_cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4))
for method in methods:
 xx=[];rr=[];tt=[]
 for n in spec['n']:
  z=[r for r in summ if r['source']=='Pharma_only' and r['method']==method and r['Pharma_fit_n']==n and r['kappa']==10];xx.append(n);rr.append(np.mean([r['mean_r'] for r in z]));tt.append(np.mean([r['E2E_saving_pct'] for r in z]))
 ax[0].plot(xx,rr,'-o',label=method);ax[1].plot(xx,tt,'-o',label=method)
ax[0].set(xlabel='Pharma Fit query budget',ylabel='Fixed Dev mean positive regret');ax[1].set(xlabel='Pharma Fit query budget',ylabel='Measured-path E2E saving, %, kappa10');ax[0].legend();fig.tight_layout();fig.savefig(C.R/'pharma_learning.png',dpi=180);plt.close(fig)
facts='Pharma合法Fit81/Dev45官方qrels及4计划结果已真实物化；126查询、1512未剖析计时、504GPU剖析；完成20/40/60/81×3seed×4同容量监督/输入×7kappa、Pharma-only/混合/旧119零样本迁移，全部曲线保留。'
facts+=' Dev最小成本且mean r<=.05的Q/QP工作点（开发选择、未认证）：'+json.dumps(C.clean(selected and [{k:x[k] for k in ['method','kappa','Dev_mean_r','Dev_mean_E2E_s']} for x in selected]),ensure_ascii=False)
decision='以固定容量、匹配风险成本而非单一预测分数判断；静态随机包络和负学习结果均保留，不提高模型容量。E3只审计功效不读Cal/Eval；当前QP成本按实测分项复用BM search，E4另测真实流式实现。'
C.prose(C.R/'E2_decision.md','# EXPLORATORY / NOT_CERTIFIED\n\n'+facts+'\n\n'+decision+'\n\nDev45由44候选查询族组成，仍无法证明IID；共享文档固定语料查询泛化，不是未见文档泛化。所有模型/阈值Dev比较均属选择过程，不可当独立效果检验。各seed不是独立query。pure zero-shot只用旧域标签调L2，完整kappa网格固定，不用Pharma结果选zero-shot参数。Page/复杂模型未训练。\n')
C.finish('PASS_LEGAL_FITDEV_TRAINING_EXPLORATORY_NOT_CERTIFIED',facts,decision)
