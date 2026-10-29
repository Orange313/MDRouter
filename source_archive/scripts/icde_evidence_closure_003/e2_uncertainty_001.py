"""Conditional paired uncertainty supplement; no fitting, retuning or new labels."""
import common as C
import numpy as np,collections,itertools,json
C.start('E2/paired_uncertainty_001');C.STAGE='E2_PAIRED_UNCERTAINTY';root=C.ART/'E2';m=C.load(root/'manifest.json','COMPLETED_E2_PRIMARY_MANIFEST');a=C.load(root/'pharma_dataset.npz','AUTHORIZED_DERIVED_FITDEV_DATA');rows=C.load(C.REPORT/'E2/per_query_policy.csv','ALREADY_COMPUTED_DEV_ROUTES');v=a['roles']=='development';ids=a['qids'][v];groups=a['family'][v];u=np.unique(groups);lookup={int(q):i for i,q in enumerate(ids)};Y=a['Y'][v];T=a['T'][v];B=2000;rng=np.random.default_rng(C.SEED);ixs=[np.concatenate([np.flatnonzero(groups==h) for h in rng.choice(u,len(u),replace=True)]) for _ in range(B)];BR=np.array([Y[ix].mean(0) for ix in ixs]);BT=np.array([T[ix].mean(0) for ix in ixs]);ref=BT[:,0]
def envelope(target):
 result=np.where(BR<=target[:,None]+1e-12,BT,np.inf).min(1)
 for j,k in itertools.combinations(range(4),2):
  denom=BR[:,k]-BR[:,j];valid=np.abs(denom)>1e-12;w=np.divide(target-BR[:,j],denom,out=np.zeros(B),where=valid);ok=valid&(w>=0)&(w<=1);cost=(1-w)*BT[:,j]+w*BT[:,k];result=np.minimum(result,np.where(ok,cost,np.inf))
 assert np.isfinite(result).all();return result
groups_rows=collections.defaultdict(list)
for r in rows:groups_rows[(r['source'],r['Pharma_fit_n'],r['seed'],r['method'],r['kappa'])].append(r)
results=[];cache={}
for key,rr in groups_rows.items():
 source,n,seed,method,kap=key;rr.sort(key=lambda r:lookup[int(r['query_id'])]);assert len(rr)==45;r=np.array([float(x['r']) for x in rr]);t=np.array([float(x['E2E_s']) for x in rr]);rb=np.array([r[ix].mean() for ix in ixs]);tb=np.array([t[ix].mean() for ix in ixs]);inc=100*(envelope(rb)-tb)/ref;ci=np.quantile(inc,[.025,.975]);results.append({'source':source,'Pharma_fit_n':n,'seed':seed,'method':method,'kappa':kap,'matched_static_random_increment_mean_bootstrap_pp':inc.mean(),'conditional_paired95_low_pp':ci[0],'conditional_paired95_high_pp':ci[1],'status':'FIXED_DEV_EXPLORATORY_CONDITIONAL_NO_REFIT','limit':'static randomized envelope reoptimized descriptively per bootstrap; no deployable gold oracle, no independence claim'})
 if source=='Pharma_only':cache[n,seed,method,kap]=(r,t)
gains=[];seeds=[str(C.SEED+i) for i in range(3)]
for method in ['Q_regret','Q_hard','Q_soft','QP_regret']:
 for kap in ['0','1','5','10','20','50','100']:
  # CSV preserves numeric spellings. Select stable numeric values, not favorable seeds.
  def find(n,s):return next(z for (nn,ss,mm,kk),z in cache.items() if int(nn)==n and int(ss)==int(s) and mm==method and float(kk)==float(kap))
  dr=np.array([find(81,s)[0]-find(20,s)[0] for s in seeds]);dt=np.array([find(81,s)[1]-find(20,s)[1] for s in seeds]);rbs=[];tbs=[]
  for ix in ixs:
   si=rng.integers(0,3,3);rbs.append(dr[si][:,ix].mean());tbs.append(-100*dt[si][:,ix].mean()/T[ix,0].mean())
  rc=np.quantile(rbs,[.025,.975]);tc=np.quantile(tbs,[.025,.975]);gains.append({'method':method,'kappa':kap,'mean_r_change_81_minus20':dr.mean(),'paired95_r_low':rc[0],'paired95_r_high':rc[1],'E2E_saving_change_81_minus20_pp':-100*dt.mean()/T[:,0].mean(),'paired95_saving_change_low_pp':tc[0],'paired95_saving_change_high_pp':tc[1],'status':'CONDITIONAL_FIXED_DEV_AND_3_TRAINING_SEED_SCENARIOS','limit':'seed resampling reflects training variation, not new independent queries; no model refitting'})
C.table(C.R/'matched_risk_uncertainty.csv',results);C.table(C.R/'learning_gain_uncertainty.csv',gains)
facts='未重训、未调参，仅对E2既有全部756组路由做配对Dev-family bootstrap2000次；同时比较20→81训练变化并加入3seed情景变异。静态/随机匹配包络逐bootstrap重算，属于开发条件诊断。'
decision='增益区间包括0时保留INSUFFICIENT；此补充不替换E2主表，不改变005或任何封存标签授权。'
C.prose(C.R/'uncertainty_scope.md','# Conditional exploratory paired intervals\n\n'+facts+'\n\n'+decision+'\n');C.finish('PASS_CONDITIONAL_PAIRED_UNCERTAINTY',facts,decision)
