import common as C
import learning_base_001 as L
import numpy as np,json
from scipy.optimize import linprog
C.start('E4');a=C.load(C.ART/'E2/pharma_dataset.npz','AUTHORIZED_FITDEV_DERIVED_DATA');v=a['roles']=='development';models=C.load(C.ART/'E2/candidate_models.json','FROZEN_CAPACITY_FIT_MODELS');f=C.load(C.ART/'E2/outcomes_001/features.json');fl={r['query_id']:r for r in f};qids=a['qids'][v];plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];selected=[];candidates=[]
for method in ['Q_regret','QP_regret']:
 choices=[]
 for m in models:
  if m['seed']!=C.SEED or m['method']!=method:continue
  model=[np.array(z) for z in m['model']];model[0]=model[0].astype(bool);X=np.c_[a['Q'][v],a['P'][v]] if method=='QP_regret' else a['Q'][v];pred=L.predict(model,X);c=np.array(m['cost_vector']);utility=c[None]+m['kappa']*np.maximum(pred,0);utility[:,1]=np.inf;act=utility.argmin(1);r=a['Y'][v][np.arange(v.sum()),act];t=a['T'][v][np.arange(v.sum()),act]+a['qwall'][v]
  if method=='QP_regret':t+=np.array([fl[int(q)]['Pstats_wall_s'] for q in qids])+np.where(act==0,np.array([fl[int(q)]['BM_search_wall_s'] for q in qids]),0)
  z={'method':method,'kappa':m['kappa'],'mean_Dev_r':float(r.mean()),'mean_Dev_E2E_s':float(t.mean()),'allocation':np.bincount(act,minlength=4).tolist(),'eligibility':bool(r.mean()<=.05)};candidates.append(z)
  if z['eligibility']:choices.append((z['mean_Dev_E2E_s'],-m['kappa'],m,z))
 if choices:
  _,_,m,z=min(choices,key=lambda x:x[:2]);selected.append({**m,'actions_allowed_indices':[0,2,3],'serving_Dev_selection':z})
C.write(C.A/'locked_models.json',selected)
domainconfigs=[]
for domain in C.DOM+['Pharmaceuticals']:
 if domain=='Pharmaceuticals':Y=a['Y'][v];T=a['T'][v]
 else:
  rows=[r for r in C.load('reports/icde_evidence_closure_003/E1/per_query_quality_cost.csv') if r['domain']==domain];ids=sorted(set(int(r['query_id']) for r in rows));lut={(int(r['query_id']),r['plan']):r for r in rows};Y=np.array([[float(lut[q,p]['r']) for p in plans] for q in ids]);T=np.array([[float(lut[q,p]['mean_E2E_s']) for p in plans] for q in ids])
 mix={};cvec=np.array(models[0]['cost_vector']) if domain=='Pharmaceuticals' else T.mean(0)/T[:,0].mean()
 if domain!='Pharmaceuticals':
  feat=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{domain}_dataset.npz');order=[list(feat['qids']).index(q) for q in ids]
 for m in selected:
  if domain=='Pharmaceuticals':risk=m['serving_Dev_selection']['mean_Dev_r']
  else:
   model=[np.array(z) for z in m['model']];model[0]=model[0].astype(bool);X=np.c_[feat['Q'][order],feat['P'][order]] if m['method']=='QP_regret' else feat['Q'][order];pred=L.predict(model,X);utility=cvec[None]+m['kappa']*np.maximum(pred,0);utility[:,1]=np.inf;act=utility.argmin(1);risk=float(Y[np.arange(len(Y)),act].mean())
  z=linprog(T.mean(0),A_ub=[Y.mean(0)],b_ub=[risk+1e-12],A_eq=[np.ones(4)],b_eq=[1],bounds=[(0,1),(0,0),(0,1),(0,1)],method='highs');assert z.success;mix['RandomMix_'+m['method']]=z.x.tolist()
 domainconfigs.append({'domain':domain,'label_free_cost_vector':cvec.tolist(),'FullV_resident_mean_s':T[:,0].mean(),'mixes':mix})
protocol={'status':'EXPLORATORY_UNCERTIFIED_POLICY_LOCKED_BEFORE_STREAM','time_bj':C.stamp(),'router_actions':['Full-Visual','BM25-R100','BM25'],'static_comparators':['BM25','BM25-R50','BM25-R100','Full-Visual'],'router_training':'PharmaFit81 only; fixed seed; fixed-Dev selection, not independent evaluation','old_domain_routing':'same Pharma-trained risk predictors transferred without label retraining; only label-free domain cost vectors change','random_mix':'query-independent draws; probabilities selected on Dev empirical static mean risk/cost; not a certificate; realized streaming risks reported','QPS':'0.5/1/2 divided by measured resident FullV mean seconds','window':[1,8,32],'concurrency':[1,2,4],'requests_per_cell':64,'input_order':'seeded random permutation of whole allowed Dev pool then repeat prefix; same sequence across methods per cell','window_semantics':'actual arrival buffering; batch feature/prediction/decision on current arrived requests; then submit paths; all waiting counted','GPU':'serialized GPU model execution lock, CPU workers can overlap; three idle checks and ResourceGuard','primary_latency':'unprofiled arrival-to-completion including window/queue/Q/P/planning, no cross-clock GPU+CPU sum','GPU_accounting':'separate W1/C1/medium-load replay profile per method; profiled latency not used as primary; unprofiled-cell GPU activity missing, not imputed','goodput':'completed requests with arrival-to-completion<=0.2s; no certified-quality goodput','matching_margin_true_mean_r':.005,'AA':'FullV same workload separate process paired replay at W1/C1/medium; >=3 process benchmark timings previously E1; stream AA new independent process','domains':domainconfigs,'all_cells_reported':True,'Cal_Eval_oldBL_test_CS_reads':0}
C.write(C.A/'serving_protocol.json',protocol);C.write(C.A/'prestream_asset_lock.json',{'protocol_SHA256':C.sha(C.A/'serving_protocol.json'),'model_SHA256':C.sha(C.A/'locked_models.json'),'code_SHA256':{p.name:C.sha(p) for p in __import__('pathlib').Path(__file__).parent.glob('*.py')}});C.table(C.R/'prestream_model_selection.csv',candidates);print(json.dumps({'status':protocol['status'],'models':[{k:m[k] for k in ['method','kappa','serving_Dev_selection']} for m in selected],'time_bj':C.stamp()}),flush=True)
