import common as C
import numpy as np,json,os,sys,collections,itertools
# Driver owns the stage. Analysis opens no new root and preserves its logs.
C.A=C.ART/'E1';C.R=C.REPORT/'E1';C.STAGE='E1'
assert C.load(C.A/'driver_completion.json')['all_successful']
plans=['BM25','BM25-R50','BM25-R100','BM25-R200','Dense-R50','Hybrid-R50','Full-Visual']
quality=C.load('reports/icde_decisive_suite_002/R2/candidate_cover_vs_quality.csv','FROZEN_OLD_DEV_EXACT_QUALITY');qlut={(r['domain'],int(r['query_id']),r['plan']):r for r in quality}
summary=[];perq=[];aa=[];fix=[];amort=[];pareto=[];allrows=[];components=[];setups=[];primary=[]
rng=np.random.default_rng(C.SEED)
for domain in C.DOM:
 features=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{domain}_dataset.npz','DEV_QUERY_FAMILY_IDS');qids=features['qids'].astype(int);groups=features['family_primary'];N=len(qids);lut={q:i for i,q in enumerate(qids)};times=np.zeros((N,3,7));cpu=np.zeros_like(times);gpu=np.zeros_like(times)
 raw=[];profiles=[]
 for rep in range(3):
  a=C.A/'workers'/f'{domain}_process_{rep:02d}';guard=C.load(a/'resource_guard.json','RESOURCE_POLLUTION_AUDIT');assert guard.get('valid',False),guard
  rr=C.load(a/'request_rows.json','NEW_ACTUAL_REQUEST_COSTS');pp=C.load(a/'profile_rows.json','NEW_ACTUAL_GPU_PROFILES');assert len(rr)==N*7*3 and len(pp)==N*7;raw+=rr;profiles+=pp
  setup=C.load(a/'setup.json');setups.append({'domain':domain,'process':rep,**setup});components+=C.load(a/'component_rows.json')
  for i,q in enumerate(qids):
   for j,p in enumerate(plans):
    z=[r for r in rr if r['query_id']==q and r['plan']==p];assert len(z)==3;times[i,rep,j]=np.mean([r['resident_E2E_s'] for r in z]);cpu[i,rep,j]=np.mean([r['CPU_s'] for r in z]);gpu[i,rep,j]=next(r['gpu_active_union_seconds'] for r in pp if r['query_id']==q and r['plan']==p)
 allrows+=raw
 # Shared query-family and process bootstrap: paired between paths; repeats are not independent n.
 u=np.unique(groups);samples=[];processes=[]
 for b in range(2000):samples.append(np.concatenate([np.flatnonzero(groups==x) for x in rng.choice(u,len(u),replace=True)]));processes.append(rng.integers(0,3,3))
 def ratio_ci(j):return np.quantile([100*(1-times[ix][:,ps,j].mean()/times[ix][:,ps,6].mean()) for ix,ps in zip(samples,processes)],[.025,.975])
 for a,b in itertools.combinations(range(3),2):
  vv=[times[ix,b,6].mean()/times[ix,a,6].mean()-1 for ix in samples];ci=np.quantile(vv,[.025,.975]);aa.append({'domain':domain,'process_A':a,'process_B':b,'relative_B_minus_A':times[:,b,6].mean()/times[:,a,6].mean()-1,'paired_family_CI_low':ci[0],'paired_family_CI_high':ci[1],'margin':.10,'status':'PASS' if ci[0]>-.10 and ci[1]<.10 else 'INSUFFICIENT_NOISE_EQUIVALENCE'})
 reliable=all(r['status']=='PASS' for r in aa if r['domain']==domain)
 for j,p in enumerate(plans):
  qq=[qlut[domain,int(q),p] for q in qids];r=np.array([float(x['r']) for x in qq]);nd=np.array([float(x['nDCG10']) for x in qq]);tt=np.array([x['resident_E2E_s'] for x in raw if x['plan']==p]);ci=ratio_ci(j);vals=np.array([r[ix].mean() for ix in samples]);ri=np.quantile(vals,[.025,.975]);saving=100*(1-times[:,:,j].mean()/times[:,:,6].mean())
  row={'domain':domain,'plan':p,'queries':N,'candidate_families':len(u),'independent_processes':3,'unprofiled_requests':N*9,'mean_nDCG10':nd.mean(),'mean_r':r.mean(),'r_CI_low':ri[0],'r_CI_high':ri[1],'harm_rate':np.mean(r>1e-10),'mean_GPU_activity_s':gpu[:,:,j].mean(),'mean_CPU_s':cpu[:,:,j].mean(),'mean_resident_E2E_s':times[:,:,j].mean(),'E2E_P50_s':np.quantile(tt,.5),'E2E_P95_s':np.quantile(tt,.95),'E2E_saving_pct':saving,'paired_E2E_saving_CI_low_pct':ci[0],'paired_E2E_saving_CI_high_pct':ci[1],'A_A_status':'PASS' if reliable else 'INSUFFICIENT','status':'SYSTEM_MEASURED_DEV_EXPLORATORY','risk_object':'mean positive nDCG10 regret to frozen FullVisual','cost_scope':'resident uninstrumented full request; setup separate; no complete business benefit claim'};summary.append(row)
  dominated=[]
  for other in plans:
   k=plans.index(other);ro=np.mean([float(qlut[domain,int(q),other]['r']) for q in qids]);to=times[:,:,k].mean()
   if ro<=r.mean()+1e-12 and to<=times[:,:,j].mean()+1e-12 and (ro<r.mean()-1e-12 or to<times[:,:,j].mean()-1e-12):dominated.append(other)
  pareto.append({**row,'dominated_by':dominated,'Pareto_scope':'Dev mean actual resident E2E and true r; descriptive point estimates'})
  for i,q in enumerate(qids):perq.append({'domain':domain,'query_id':int(q),'family':groups[i],'plan':p,**{k:qlut[domain,int(q),p][k] for k in ['nDCG10','r','signed_gap','reference_top10_full','reference_positive_top10_missing','gold_recall_at_candidate_k']},'mean_GPU_activity_s':gpu[i,:,j].mean(),'mean_CPU_s':cpu[i,:,j].mean(),'mean_E2E_s':times[i,:,j].mean(),'E2E_process_means':times[i,:,j].tolist()})
  # Joint setup includes both dense and visual models. It is conservative for non-dense arms, not arm-specific measured setup.
  setupmean=np.mean([s['total_joint_cold_setup_wall_s'] for s in setups if s['domain']==domain]);indexid=dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']))[domain];im=C.load(f'cache/indexes/{indexid}/index_manifest.json','ORIGINAL_INDEX_BUILD_METADATA')
  for Q in [1000,10000,100000,1000000]:amort.append({'domain':domain,'plan':p,'Q':Q,'resident_mean_E2E_s':times[:,:,j].mean(),'shared_joint_residency_setup_wall_s':setupmean,'original_visual_index_build_wall_s':im.get('visual_seconds'),'original_dense_index_build_wall_s':im.get('dense_seconds'),'shared_setup_amortized_E2E_s':times[:,:,j].mean()+setupmean/Q,'shared_setup_saving_vs_V_pct':100*(1-(times[:,:,j].mean()+setupmean/Q)/(times[:,:,6].mean()+setupmean/Q)),'shared_index_GPU_build_s':None,'cost_status':'shared setup residency bound; original index build wall separately; per-plan setup GPU missing, no complete net claim'})
 for q in qids:
  b=qlut[domain,int(q),'BM25-R50'];z=qlut[domain,int(q),'BM25-R100'];fix.append({'domain':domain,'query_id':int(q),'r_R50':float(b['r']),'r_R100':float(z['r']),'R50_harm':float(b['r'])>1e-10,'R100_repaired_to_zero':float(b['r'])>1e-10 and float(z['r'])<=1e-10,'positive_missing_R50':b['reference_positive_top10_missing'],'positive_missing_R100':z['reference_positive_top10_missing']})
 harms=[x for x in fix if x['domain']==domain and x['R50_harm']];repair=np.mean([x['R100_repaired_to_zero'] for x in harms]) if harms else None;s=next(x for x in summary if x['domain']==domain and x['plan']=='BM25-R100');primary.append({'domain':domain,'R50_harmed_queries':len(harms),'R100_zero_harm_repairs':sum(x['R100_repaired_to_zero'] for x in harms),'repair_fraction':repair,'major_repair_status':'PASS' if repair is not None and repair>=.5 else 'FAIL','R100_E2E_saving_pct':s['E2E_saving_pct'],'paired95_low':s['paired_E2E_saving_CI_low_pct'],'paired95_high':s['paired_E2E_saving_CI_high_pct'],'cost_advantage_status':'PASS' if reliable and s['E2E_saving_pct']>=10 and s['paired_E2E_saving_CI_low_pct']>0 else 'INSUFFICIENT' if not reliable else 'FAIL'})
C.table(C.R/'static_quality_e2e.csv',summary);C.table(C.R/'k_sweep_pareto.csv',pareto);C.table(C.R/'candidate_miss_fix.csv',fix);C.table(C.R/'setup_amortization.csv',amort);C.table(C.R/'aa_controls.csv',aa);C.table(C.R/'per_query_quality_cost.csv',perq);C.table(C.R/'all_unprofiled_requests.csv',allrows);C.table(C.R/'GPU_component_profiles.csv',components);C.table(C.R/'process_setup.csv',setups);C.write(C.A/'primary_decisions.json',primary)
sys.path.insert(0,str(C.ROOT/'cache/report_dependencies/prereg_v2_round_001'));os.environ['MPLCONFIGDIR']=str(C.A/'matplotlib_cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
for d in C.DOM:
 fig,ax=plt.subplots(figsize=(7,4))
 for row in summary:
  if row['domain']==d:ax.scatter(row['mean_resident_E2E_s'],row['mean_r']);ax.annotate(row['plan'],(row['mean_resident_E2E_s'],row['mean_r']),fontsize=7)
 ax.set(xlabel='Measured resident full-request mean E2E seconds',ylabel='Dev true mean positive nDCG10 regret',title=d+'; 3 independent processes, 9 timings/query/plan');fig.tight_layout();fig.savefig(C.R/(d+'_cost_risk.png'),dpi=180);plt.close(fig)
facts='全119 Dev、7路径、3独立进程/域完成7497未剖析请求计时与2499全路径GPU剖析，所有Top10精确一致；'+json.dumps(C.clean(primary),ensure_ascii=False)
decision='只根据真实成本与质量选择最小动作；R100未修复的伤害、被支配路径和A/A噪声均保留。进入E2合法PharmaFit/Dev数据学习曲线；不得提升模型容量或调用Cal/Eval。'
C.prose(C.R/'E1_decision.md','# SYSTEM_MEASURED / DEV_EXPLORATORY\n\n'+facts+'\n\n'+decision+'\n\n限制：配对family+process bootstrap2000次、逐点区间，Dev条件描述；family不证明IID。GPU剖析与未剖析E2E独立计时队列。CPU主机共享、setup分项有missing，未报告完整业务净收益。Hybrid是否值得额外成本按k_sweep_pareto.csv全表判断。所有固定路径已含自身Probe，路由附加Probe/规划成本在E2/E4实际支付。\n')
C.finish('PASS_MEASUREMENTS_EXPLORATORY_SCIENTIFIC_DECISIONS',facts,decision)
