import common as C
import frontier_math as F
import learning_base_001 as L
import numpy as np,json,collections,hashlib
from scipy.stats import spearmanr
C.start('S3');a=C.load(C.PREV/'E2/pharma_dataset.npz');rows=C.load(C.PR/'E2/per_query_policy.csv');summ=C.load(C.PR/'E2/learning_curve_pharma.csv');qs=C.load(C.PREV/'E2/allowed_fitdev_queries.json');qrels=C.load(C.PREV/'E2/public_projection_final_001/fitdev_qrels.json','AUTHORIZED_FITDEV_ONLY');pages=C.load('cache/indexes/pharma_fixedcorpus_index_001/pages.json','UNLABELED_FIXED_CORPUS');pd={int(p['corpus_id']):p['doc_id'] for p in pages};qd={int(q['query_id']):q for q in qs};docs=collections.defaultdict(set);reln=collections.Counter()
for r in qrels:docs[int(r['query_id'])].add(pd[int(r['corpus_id'])]);reln[int(r['query_id'])]+=1
ids=a['qids'].astype(int);g=a['family'];fit=np.flatnonzero(a['roles']=='fit');dev=np.flatnonzero(a['roles']=='development');assert len(fit)==81 and len(dev)==45;plans=C.PLANS;Y=a['Y'];T=a['T'];lookup={int(q):i for i,q in enumerate(ids[dev])};methods=['Q_regret','QP_regret'];seeds=sorted({int(r['seed']) for r in summ if r['source']=='Pharma_only'});ks=[0,1,5,10,20,50,100];ns=[20,40,60,81];cal=[];comp=[];subsets=[];nest=[];samples={};predicted={};binstats=[];verified=[]
def take(n,seed):
 order=sorted(set(g[fit]),key=lambda x:hashlib.sha256((str(seed)+x).encode()).hexdigest());chosen=[]
 for h in order:
  ix=fit[g[fit]==h]
  if len(chosen)+len(ix)<=n:chosen.extend(ix.tolist())
 assert len(chosen)==n;return np.array(chosen)
for seed in seeds:
 for n in ns:
  tr=take(n,seed);samples[seed,n]=tr;dd=set().union(*(docs[int(q)] for q in ids[tr]));comp.append({'seed':seed,'Fit_n':n,'candidate_families':len(set(g[tr])),'distinct_source_documents':len(dd),'mean_relevant_pages':np.mean([reln[int(q)] for q in ids[tr]]),'generator_composition':dict(collections.Counter(qd[int(q)].get('generator','UNKNOWN') for q in ids[tr])),'BM25_harm_fraction':np.mean(Y[tr,3]>1e-10),'BM25_r_gt05_fraction':np.mean(Y[tr,3]>.05),'R50_harm_fraction':np.mean(Y[tr,1]>1e-10),'R100_harm_fraction':np.mean(Y[tr,2]>1e-10),'R100_r_gt05_fraction':np.mean(Y[tr,2]>.05),'mean_r_BM25':Y[tr,3].mean(),'mean_r_R100':Y[tr,2].mean(),'query_ids':ids[tr].tolist(),'query_id_SHA256':hashlib.sha256(json.dumps(ids[tr].tolist()).encode()).hexdigest(),'status':'FIT_COMPOSITION_NOT_IID_EVIDENCE'})
  for i in tr:subsets.append({'seed':seed,'requested_n':n,'query_id':int(ids[i]),'family':g[i],'source_docs':sorted(docs[int(ids[i])]),'BM25_r':Y[i,3],'R100_r':Y[i,2],'role':'FIT_ONLY'})
 for n1,n2 in zip(ns[:-1],ns[1:]):
  s1=set(ids[samples[seed,n1]]);s2=set(ids[samples[seed,n2]]);nest.append({'seed':seed,'smaller_n':n1,'larger_n':n2,'smaller_is_subset':s1<=s2,'retained_n':len(s1&s2),'removed_query_ids':sorted(s1-s2),'added_n':len(s2-s1),'status':'NESTED' if s1<=s2 else 'PAIRED_FIXED_DEV_BUT_NOT_NESTED'})
rr=collections.defaultdict(list)
for r in rows:
 if r['source']=='Pharma_only':rr[int(r['Pharma_fit_n']),int(r['seed']),r['method'],float(r['kappa'])].append(r)
for n in ns:
 for seed in seeds:
  tr=samples[seed,n]
  for method in methods:
   X=np.c_[a['Q'],a['P']] if method=='QP_regret' else a['Q']
   for kap in ks:
    z=rr[n,seed,method,float(kap)];z.sort(key=lambda r:lookup[int(r['query_id'])]);pred=np.array([json.loads(r['prediction']) for r in z]);act=np.array([plans.index(r['action']) for r in z]);risk=np.array([float(r['r']) for r in z]);time=np.array([float(r['E2E_s']) for r in z]);s=next(r for r in summ if r['source']=='Pharma_only' and int(r['Pharma_fit_n'])==n and int(r['seed'])==seed and r['method']==method and float(r['kappa'])==kap);lam=float(s['L2']);m=L.fit(X[tr],Y[tr],lam);again=L.predict(m,X[dev]);err=float(np.max(np.abs(pred-again)));assert err<1e-9,(n,seed,method,kap,err);verified.append({'n':n,'seed':seed,'method':method,'kappa':kap,'L2_trainselected':lam,'predictions_reproduced_max_error':err,'status':'PASS_NO_RESELECTION'});predicted[n,seed,method,kap]=(risk,time,pred,act)
    cols,med,mu,sd,w,bias=m;xd=X[dev][:,cols];xd=np.c_[xd,np.zeros_like(xd)];std=(xd-mu)/sd;drift=np.median(np.linalg.norm(std,axis=1));bad=(act!=0)&(risk>.05);selected_pred=pred[np.arange(len(dev)),act];vtrue=Y[dev];den=np.sum((vtrue-vtrue.mean(0))**2)
    for j in [1,2,3]:
     yp=pred[:,j];yt=vtrue[:,j];corr=spearmanr(yp,yt).statistic if np.std(yt)>0 and np.std(yp)>0 else None;denj=np.sum((yt-yt.mean())**2);cal.append({'Fit_n':n,'seed':seed,'method':method,'kappa':kap,'head':plans[j],'L2':lam,'Dev_n':45,'Dev_families':44,'true_mean_r':yt.mean(),'predicted_mean_r':yp.mean(),'mean_prediction_bias_pred_minus_true':np.mean(yp-yt),'MAE':np.mean(abs(yp-yt)),'MSE':np.mean((yp-yt)**2),'R2':1-np.sum((yp-yt)**2)/denj if denj else None,'Spearman':corr,'pred_negative_fraction':np.mean(yp<0),'Dev_std_feature_norm_median':drift,'mean_selected_true_r':risk.mean(),'mean_selected_pred_r':selected_pred.mean(),'selected_calibration_bias':np.mean(selected_pred-risk),'false_safe_n':int(bad.sum()),'false_safe_regret_fraction':risk[bad].sum()/risk.sum() if risk.sum()>0 else 0,'allocation':np.bincount(act,minlength=4).tolist(),'E2E_s':time.mean(),'scope':'original four-action E2 learning, fixed Dev, unchanged trainselected L2; not S2 three-action serving catalog'})
    for low,high in zip([-np.inf,0,.01,.025,.05,.1,.25],[0,.01,.025,.05,.1,.25,np.inf]):
     mask=(selected_pred>=low)&(selected_pred<high)
     binstats.append({'Fit_n':n,'seed':seed,'method':method,'kappa':kap,'bin_low':low if np.isfinite(low) else '-inf','bin_high':high if np.isfinite(high) else 'inf','n':int(mask.sum()),'pred_mean':selected_pred[mask].mean() if mask.any() else None,'true_mean':risk[mask].mean() if mask.any() else None,'false_safe_n':int(np.sum(mask&bad)),'status':'DEV_CALIBRATION_DIAGNOSTIC_NO_NEW_THRESHOLD'})
ixs=F.boots(g[dev],2000,C.SEED);BR=np.array([Y[dev][ix].mean(0) for ix in ixs]);BT=np.array([T[dev][ix].mean(0) for ix in ixs]);rng=np.random.default_rng(C.SEED);si=rng.integers(0,3,(2000,3));gains=[];matched=[]
for method in methods:
 for kap in ks:
  results={}
  for n in ns:
   rr0=np.array([predicted[n,s,method,kap][0] for s in seeds]);tt0=np.array([predicted[n,s,method,kap][1] for s in seeds]);br=np.array([[r[ix].mean() for ix in ixs] for r in rr0]);bt=np.array([[t[ix].mean() for ix in ixs] for t in tt0]);ev=np.array([F.envelope(BR,BT,r)[0] for r in br]);inc=100*(ev-bt)/BT[:,0][None];point=np.mean([100*(F.envelope(Y[dev].mean(0),T[dev].mean(0),r.mean())[0][0]-t.mean())/T[dev,0].mean() for r,t in zip(rr0,tt0)]);results[n]=(rr0,tt0,br,bt,inc,point)
   ci=np.quantile(np.mean(np.take_along_axis(inc.T,si,axis=1),axis=1),[.025,.975]);matched.append({'method':method,'kappa':kap,'Fit_n':n,'mean_Dev_r':rr0.mean(),'mean_E2E_s':tt0.mean(),'matched_risk_increment_pp':point,'paired95_low_pp':ci[0],'paired95_high_pp':ci[1],'interval':'same fixed Dev candidate families + three paired seed scenarios; not independent generalization'})
  r20,t20,br20,bt20,i20,p20=results[20];r81,t81,br81,bt81,i81,p81=results[81];gain=np.mean(np.take_along_axis((i81-i20).T,si,axis=1),axis=1);rchange=np.mean(np.take_along_axis((br81-br20).T,si,axis=1),axis=1);ci=np.quantile(gain,[.025,.975]);ri=np.quantile(rchange,[.025,.975]);gains.append({'method':method,'kappa':kap,'risk20':r20.mean(),'risk81':r81.mean(),'risk_change81_minus20':r81.mean()-r20.mean(),'paired95_r_change_low':ri[0],'paired95_r_change_high':ri[1],'matched_risk_increment20_pp':p20,'matched_risk_increment81_pp':p81,'matched_risk_improvement81_minus20_pp':p81-p20,'paired95_matched_improvement_low_pp':ci[0],'paired95_matched_improvement_high_pp':ci[1],'status':'FIXED_DEV_ASSOCIATIONAL_NOT_CAUSAL'})
C.table(C.R/'paired_fit_subsets.csv',subsets);C.table(C.R/'subset_composition.csv',comp);C.table(C.R/'nestedness_checks.csv',nest);C.table(C.R/'risk_calibration_by_n.csv',cal);C.table(C.R/'calibration_bins.csv',binstats);C.table(C.R/'prediction_reproduction.csv',verified);C.table(C.R/'matched_risk_learning_curves.csv',matched);C.table(C.R/'paired_learning_gain_matched_risk.csv',gains)
plt=C.plots();fig,axes=plt.subplots(2,2,figsize=(11,7))
for mi,method in enumerate(methods):
 for kap in ks:
  r=[x for x in matched if x['method']==method and x['kappa']==kap];axes[0,mi].plot([z['Fit_n'] for z in r],[z['mean_Dev_r'] for z in r],'-o',markersize=3,label=f'kappa={kap}');axes[1,mi].plot([z['Fit_n'] for z in r],[z['matched_risk_increment_pp'] for z in r],'-o',markersize=3)
 axes[0,mi].set(title=method,ylabel='Fixed Dev mean positive regret');axes[1,mi].set(xlabel='Fit query count',ylabel='Matched static-random increment (pp)');axes[1,mi].axhline(0,color='gray',lw=.7)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=8);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(C.R/'learning_curves_all_kappa.png',dpi=180);fig.savefig(C.R/'learning_curves_all_kappa.pdf');plt.close(fig)
C.write(C.A/'analysis_complete.json',{'time_bj':C.stamp(),'status':'PASS_MINIMAL_DIAGNOSTIC','prediction_checks':len(verified),'reproduced_max_error':max(z['predictions_reproduced_max_error'] for z in verified),'nestedness':nest,'n20_81':gains,'new_Cal_Eval_reads':0});print(json.dumps({'status':'S3_COMPUTED','prediction_checks':len(verified),'kappa10':[z for z in gains if z['kappa']==10],'nested_all':all(z['smaller_is_subset'] for z in nest),'time_bj':C.stamp()}),flush=True)
