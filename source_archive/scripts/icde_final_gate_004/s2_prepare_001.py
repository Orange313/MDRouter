import common as C
import learning_base_001 as L
import engine_001 as E
import numpy as np,ast,types,collections,math,sys,time,json,os
from scipy.optimize import linprog
C.start('S2');os.sched_setaffinity(0,[6,7,26,27]);sys.path.insert(0,str(C.ROOT/'src'))
from aaber.router.fast_query_router import FastLexicalFeatureEncoder
from aaber.common.text import tokenize
from mdrouter_v5.graded_metrics import stable_ranking
source=C.load('scripts/cross_domain_four_plan_001.py');node=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='BM25');env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'np':np,'Counter':collections.Counter,'defaultdict':collections.defaultdict,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module([node],type_ignores=[]),'frozen_BM25', 'exec'),env)
dc={x['domain']:x for x in C.load(C.ART/'S0/domain_policy_config.json')};models=C.load(C.ART/'S0/locked_models.json');mm={}
for m in models:z=[np.array(v) for v in m['model']];z[0]=z[0].astype(bool);mm[m['method']]=(m,z)
e1=C.load(C.PR/'E1/per_query_quality_cost.csv');ph=C.load(C.PREV/'E2/outcomes_001/outcomes.json');records=[];featrows=[];paths=[];policy=[];oracles=[];costsummary=[]
for di,d in enumerate(C.DOM):
 ix=['hr_primary_001','finance_primary_001','industrial_primary_003','pharma_fixedcorpus_index_001'][di];pages=C.load(f'cache/indexes/{ix}/pages.json','UNLABELED_FIXED_CORPUS');indexmeta=C.load(f'cache/indexes/{ix}/index_manifest.json','UNLABELED_INDEX_METADATA');enc=FastLexicalFeatureEncoder();enc.fit_corpus([types.SimpleNamespace(searchable_text=p['markdown'] or '') for p in pages]);bm=env['BM25'](pages)
 if d=='Pharmaceuticals':
  qs=[q for q in C.load(C.PREV/'E2/allowed_fitdev_queries.json') if q['role']=='development'];data=C.load(C.PREV/'E2/pharma_dataset.npz');v=data['roles']=='development';qids=data['qids'][v].astype(int);families=data['family'][v];cachedQ=data['Q'][v];cachedP=data['P'][v];rows=[r for r in ph if r['role']=='development'];lut={(int(r['query_id']),r['plan']):r for r in rows};T=np.array([[r['mean_E2E_s'] for p in C.PLANS for r in [lut[q,p]]] for q in qids]);G=np.array([[lut[q,p]['GPU_activity_s'] for p in C.PLANS] for q in qids]);CPU=np.array([[lut[q,p]['mean_CPU_s'] for p in C.PLANS] for q in qids])
 else:
  qs=C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json');data=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz');qids=data['qids'].astype(int);families=data['family_primary'];cachedQ=data['Q'];cachedP=data['P'];rows=[r for r in e1 if r['domain']==d];lut={(int(r['query_id']),r['plan']):r for r in rows};T=np.array([[float(lut[q,p]['mean_E2E_s']) for p in C.PLANS] for q in qids]);G=np.array([[float(lut[q,p]['mean_GPU_activity_s']) for p in C.PLANS] for q in qids]);CPU=np.array([[float(lut[q,p]['mean_CPU_s']) for p in C.PLANS] for q in qids])
 Y=np.array([[float(lut[q,p]['r']) for p in C.PLANS] for q in qids]);U=np.array([[float(lut[q,p]['nDCG10']) for p in C.PLANS] for q in qids]);ql={int(q['query_id']):q for q in qs};n=len(qids);Q=[];P=[];over=[]
 for qi,qid in enumerate(qids):
  q=ql[qid]['query'];qwall=[];qcpu=[];bwall=[];bcpu=[];pwall=[];pcpu=[]
  for rep in range(3):
   ct=time.thread_time();t=time.perf_counter();qq=enc.transform(types.SimpleNamespace(question=q));x=np.array([qq[k] for k in enc.feature_names]);qwall.append(time.perf_counter()-t);qcpu.append(time.thread_time()-ct);ct=time.thread_time();t=time.perf_counter();probe=bm.search(q);bwall.append(time.perf_counter()-t);bcpu.append(time.thread_time()-ct);ct=time.thread_time();t=time.perf_counter();p=E.probe_features(bm,q,*probe);pwall.append(time.perf_counter()-t);pcpu.append(time.thread_time()-ct)
  assert np.allclose(x,cachedQ[qi],rtol=1e-5,atol=1e-6) and np.allclose(p,cachedP[qi],rtol=1e-5,atol=1e-6);Q.append(x);P.append(p);ov={k:float(np.mean(z)) for k,z in zip(['Q_wall','Q_CPU','BM_wall','BM_CPU','Pstats_wall','Pstats_CPU'],[qwall,qcpu,bwall,bcpu,pwall,pcpu])};over.append(ov);featrows.append({'domain':d,'query_id':int(qid),'family':families[qi],'Q_difference_max':np.max(abs(x-cachedQ[qi])),'P_difference_max':np.max(abs(p-cachedP[qi])),**ov,'repeats':3,'scope':'new CPU-only timing; Q/P semantic match; no GPU/labels used for feature extraction'})
 Q=np.array(Q);P=np.array(P);pol={};preds={};acts={};j=np.arange(n)
 for p in C.PLANS:
  a=C.PLANS.index(p);w=np.eye(4)[np.repeat(a,n)];pol[p]={'weights':w,'r':Y[:,a],'U':U[:,a],'t':T[:,a],'g':G[:,a],'cpu':CPU[:,a],'act':np.repeat(a,n),'planning_wall':np.zeros(n)}
 for name,(m,model) in mm.items():
  X=np.c_[Q,P] if name=='QP_regret' else Q;pred=L.predict(model,X);uv=np.array(dc[d]['label_free_cost_vector'])[None]+m['kappa']*np.maximum(pred,0);uv[:,1]=np.inf;act=uv.argmin(1);pw=[];pc=[]
  for x in X:
   ts=[];cs=[]
   for _ in range(3):
    ct=time.thread_time();t=time.perf_counter();pr=L.predict(model,x[None]);utility=np.array(dc[d]['label_free_cost_vector'])[None]+m['kappa']*np.maximum(pr,0);utility[:,1]=np.inf;utility.argmin(1);ts.append(time.perf_counter()-t);cs.append(time.thread_time()-ct)
   pw.append(np.mean(ts));pc.append(np.mean(cs))
  extra=np.array([o['Q_wall'] for o in over])+pw;exc=np.array([o['Q_CPU'] for o in over])+pc
  if name=='QP_regret':extra+=np.array([o['Pstats_wall'] for o in over])+np.where(act==0,[o['BM_wall'] for o in over],0);exc+=np.array([o['Pstats_CPU'] for o in over])+np.where(act==0,[o['BM_CPU'] for o in over],0)
  pol[name]={'weights':np.eye(4)[act],'r':Y[j,act],'U':U[j,act],'t':T[j,act]+extra,'g':G[j,act],'cpu':CPU[j,act]+exc,'act':act,'planning_wall':np.array(pw)};acts[name]=act;preds[name]=pred
  for i,qid in enumerate(qids):policy.append({'domain':d,'method':name,'query_id':int(qid),'family':families[i],'action':C.PLANS[act[i]],'prediction':pred[i].tolist(),'r':Y[i,act[i]],'nDCG10':U[i,act[i]],'E2E_composed_s':pol[name]['t'][i],'GPU_activity_s':G[i,act[i]],'CPU_composed_s':pol[name]['cpu'][i],'false_safe_r_gt05':bool(act[i]!=0 and Y[i,act[i]]>.05),'predicted_selected_r':float(pred[i,act[i]]),'cost_scope':'prior measured resident path + newly measured sequential Q/P/decision CPU wall; COMPOSED not a new actual stream; Probe reused on BM paths; FullV pays extra Probe'})
 for name,w in dc[d]['mixes'].items():
  w=np.array(w);pol[name]={'weights':np.repeat(w[None],n,axis=0),'r':Y@w,'U':U@w,'t':T@w,'g':G@w,'cpu':CPU@w,'act':np.repeat(-1,n),'planning_wall':np.zeros(n)}
 np.savez(C.A/(d+'_matrix.npz'),qids=qids,families=families,Y=Y,U=U,T=T,G=G,CPU=CPU,Q=Q,P=P,**{'r_'+k:v['r'] for k,v in pol.items()},**{'t_'+k:v['t'] for k,v in pol.items()},**{'U_'+k:v['U'] for k,v in pol.items()},**{'g_'+k:v['g'] for k,v in pol.items()},**{'cpu_'+k:v['cpu'] for k,v in pol.items()},**{'act_'+k:v['act'] for k,v in pol.items()})
 for name,v in pol.items():costsummary.append({'domain':d,'method':name,'Dev_n':n,'candidate_families':len(set(families)),'mean_r':v['r'].mean(),'mean_nDCG10':v['U'].mean(),'resident_or_composed_mean_E2E_s':v['t'].mean(),'GPU_active_s':v['g'].mean(),'CPU_s':v['cpu'].mean(),'planning_wall_s':v['planning_wall'].mean(),'scope':'COMPOSED_FULL_PATH_DEV' if name in mm else 'MEASURED_STATIC_OR_FROZEN_MIX_EXPECTATION','new_stream_status':'PENDING_S1_GATE','certification':'NOT_CERTIFIED'})
 C.write(C.A/(d+'_policy_costs.json'),{k:{'weights':v['weights'].tolist(),'mean_r':float(v['r'].mean())} for k,v in pol.items()});C.write(C.A/(d+'_setup_metadata.json'),{'index_manifest':indexmeta,'prior_residency_setup':C.load(C.PREV/f'E4/workers/{d}_main_001/setup.json'),'Q_assumption':'separate setup/Q accounting; deployment Q not frozen','missing':['per-plan original setup GPU if absent','model-load GPU activity','network','full host energy']})
 print(json.dumps({'stage':'S2_CPU_PREPARED','domain':d,'Dev_n':n,'Q_r':float(pol['Q_regret']['r'].mean()),'QP_r':float(pol['QP_regret']['r'].mean()),'time_bj':C.stamp()}),flush=True)
C.table(C.R/'CPU_feature_costs.csv',featrows);C.table(C.R/'composed_path_quality_cost.csv',costsummary);C.table(C.R/'per_query_frozen_policy.csv',policy);C.write(C.A/'preparation_complete.json',{'time_bj':C.stamp(),'status':'PASS','note':'no refit, unchanged policies/costvector, only authorized Dev/Fit derived inputs; composed paths are not new stream measurements','protected_reads':0})
