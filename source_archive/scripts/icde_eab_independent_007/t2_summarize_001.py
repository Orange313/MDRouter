import common as C
import sys,collections,json,time
import numpy as np
from scipy.optimize import linprog
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
C.start('T2/summary_001')
protocol=C.load(C.ART/'T1/biomedical_frozen_protocol.json','FROZEN_BIOMEDICAL_PROTOCOL');B=2000
quality=[];raw=[];profiles=[]
for p in range(3):
    guard=C.load(C.ART/f'T2/process_{p}_001/resource_guard.json','RESOURCE_GUARD_RECEIPT');assert guard.get('valid') is True and not guard.get('foreign_pids') and not guard.get('errors'),'RESOURCE_GUARD_INVALID'
    a=C.load(C.REPORT/f'T2/process_{p}_001/quality.csv','NEW_BIOMEDICAL_FIXED_POLICY_OUTCOMES')
    if p==0:quality=a
    else:
        for x,y in zip(quality,a):assert all(x[k]==y[k] for k in ['query_id','method','action','ranking','nDCG10','r'])
    raw += C.load(C.REPORT/f'T2/process_{p}_001/requests.csv','NEW_HARDWARE_TIMINGS')
profiles=C.load(C.REPORT/'T2/process_0_001/GPU_profiles.csv','NEW_SEPARATE_GPU_PROFILES')
methods=C.PLANS+['Q_regret','QP_regret','RandomMix','RR_hard','RR_soft'];ids=protocol['query_ids'];N=len(ids)
qmap={(int(r['query_id']),r['method']):r for r in quality};by=collections.defaultdict(list)
for r in raw:by[int(r['query_id']),int(r['process']),r['method']].append(r)
T=np.array([[[np.mean([float(r['E2E_s']) for r in by[q,p,m]]) for m in methods] for p in range(3)] for q in ids]);CPU=np.array([[[np.mean([float(r['CPU_process_s']) for r in by[q,p,m]]) for m in methods] for p in range(3)] for q in ids])
Y=np.array([[float(qmap[q,m]['r']) for m in methods] for q in ids]);U=np.array([[float(qmap[q,m]['nDCG10']) for m in methods] for q in ids]);per=[];main=[];boots=[];matched=[];front=[];gap=[]
for i,q in enumerate(ids):
    for mi,m in enumerate(methods):
        pp=[r for r in profiles if int(r['query_id'])==q and r['method']==m and r['component']=='complete_route'];rows=[r for p in range(3) for r in by[q,p,m]];a=qmap[q,m]
        per.append({**a,'E2E_mean_s':T[i,:,mi].mean(),'E2E_P50_s':np.median([float(r['E2E_s']) for r in rows]),'E2E_P95_s':np.quantile([float(r['E2E_s']) for r in rows],.95),'CPU_process_s':CPU[i,:,mi].mean(),'GPU_active_s':np.mean([float(r['gpu_active_union_seconds']) for r in pp]) if pp else None,'GPU_cost_scope':'new separate instrumented complete route; not latency cohort','processes':3,'timed_repeats_per_process':3,'E2E_scope':'actual complete warm route; single-request residency, not Serving'})
for grouping in ['family','source_group']:
    groups=np.array([qmap[q,'Full-Visual'][grouping] for q in ids]);ix=F.boots(groups,B,C.SEED+(0 if grouping=='family' else 1));rng=np.random.default_rng(C.SEED+43);pi=rng.integers(0,3,(B,3));bt=np.array([T[i][:,p].mean((0,1)) for i,p in zip(ix,pi)]);byy=np.array([Y[i].mean(0) for i in ix]);bu=np.array([U[i].mean(0) for i in ix])
    for mi,m in enumerate(methods):
        for measure,arr in [('mean_r',byy[:,mi]),('nDCG10',bu[:,mi]),('E2E_ms',1000*bt[:,mi]),('E2E_saving_pp',100*(1-bt[:,mi]/bt[:,0]))]:
            lo,hi=np.quantile(arr,[.025,.975]);boots.append({'method':m,'grouping':grouping,'groups':len(set(groups)),'metric':measure,'low95':lo,'high95':hi,'bootstrap':2000,'interpretation':'pointwise paired cluster + process bootstrap; conditional descriptive, no IID or PAC claim'})
        ev,w=F.envelope(Y[:,:4].mean(0),T[:,:,:4].mean((0,1)),Y[:,mi].mean());eb,_=F.envelope(byy[:,:4],bt[:,:4],byy[:,mi]);delta=100*(eb-bt[:,mi])/bt[:,0];lo,hi=np.quantile(delta,[.025,.975]);matched.append({'method':m,'grouping':grouping,'mean_r':Y[:,mi].mean(),'actual_policy_E2E_ms':T[:,:,mi].mean()*1000,'static_random_envelope_ms':ev[0]*1000,'increment_vs_envelope_pp_of_FullV':100*(ev[0]-T[:,:,mi].mean())/T[:,:,0].mean(),'paired95_low_pp':lo,'paired95_high_pp':hi,'allocation':w[0].tolist(),'status':'RETROSPECTIVE_DESCRIPTIVE_NOT_A_DEPLOYED_POLICY'})
        for control in ['RandomMix','BM25-R100']:
            j=methods.index(control)
            for measure,arr,point in [('r_difference',byy[:,mi]-byy[:,j],Y[:,mi].mean()-Y[:,j].mean()),('E2E_difference_ms',1000*(bt[:,mi]-bt[:,j]),1000*(T[:,:,mi].mean()-T[:,:,j].mean()))]:
                lo,hi=np.quantile(arr,[.025,.975]);boots.append({'method':m,'control':control,'grouping':grouping,'metric':measure,'estimate':point,'low95':lo,'high95':hi,'groups':len(set(groups)),'bootstrap':B,'interpretation':'paired descriptive; frozen policies, no tuning'})
    for eps in [0,.01,.02,.05,.1,.2]:
        ev,w=F.envelope(Y[:,:4].mean(0),T[:,:,:4].mean((0,1)),eps);eb,_=F.envelope(byy[:,:4],bt[:,:4],np.full(B,eps));lo,hi=np.quantile(100*(1-eb/bt[:,0]),[.025,.975]);front.append({'risk':eps,'grouping':grouping,'cost_ms':1000*ev[0],'allocation':w[0].tolist(),'saving_pp':100*(1-ev[0]/T[:,:,0].mean()),'paired95_low_pp':lo,'paired95_high_pp':hi,'status':'RETROSPECTIVE_DESCRIPTIVE_IDEAL_STATIC_RANDOM_FRONTIER'})
for mi,m in enumerate(methods):
    rr=[r for r in raw if r['method']==m];qq=[r for r in per if r['method']==m];ci=[r for r in boots if r['method']==m and r['grouping']=='source_group' and 'control' not in r]
    row={'dataset':'Biomedical Lectures V2 English','retriever':'ColQwen2-v1.0-merged','method':m,'n_queries':160,'n_query_families':len({r['family'] for r in qq}),'n_sources':27,'nDCG10':U[:,mi].mean(),'mean_r':Y[:,mi].mean(),'harm_rate':(Y[:,mi]>1e-10).mean(),'harm_severity_p95':np.quantile(Y[:,mi],.95),'r_bar':np.mean([float(r['r_bar']) for r in qq]),'E2E_mean_ms':T[:,:,mi].mean()*1000,'E2E_P50_ms':1000*np.median([float(r['E2E_s']) for r in rr]),'E2E_P95_ms':1000*np.quantile([float(r['E2E_s']) for r in rr],.95),'CPU_process_ms':CPU[:,:,mi].mean()*1000,'GPU_active_s':np.mean([r['GPU_active_s'] for r in qq]),'Probe_extra_ms':1000*np.mean([float(r['Probe_extra_wall_s'])+float(r['Probe_statistics_wall_s']) for r in rr]),'Q_features_ms':1000*np.mean([float(r['Q_wall_s']) for r in rr]),'planner_ms':1000*np.mean([float(r['planner_wall_s']) for r in rr]),'risk_scope':'positive regret relative model-specific FullV; NOT_CERTIFIED','cost_scope':'NEW_ACTUAL_COMPLETE_WARM_PATH; separate instrumented GPU; not Serving or business net benefit','evidence_level':'PREREGISTERED_FROZEN_POLICY_DESCRIPTIVE; pretraining overlap UNKNOWN','baseline_fidelity':'ADAPTED_NOT_FAITHFUL' if m.startswith('RR') else 'FIXED_LOCAL_BASELINE','status':'PASS_EXECUTED'}
    for z in ci:row[z['metric']+'_source95_low']=z['low95'];row[z['metric']+'_source95_high']=z['high95']
    main.append(row)
for eps in [0,.01,.02,.05,.1,.2]:
    cat=[0,2,3];yy=Y[:,cat];tt=T.mean(1)[:,cat];lp=linprog(tt.ravel(),A_ub=[yy.ravel()],b_ub=[eps*N],A_eq=np.kron(np.eye(N),np.ones((1,3))),b_eq=np.ones(N),bounds=(0,1),method='highs');assert lp.success;ev,_=F.envelope(yy.mean(0),tt.mean(0),eps)
    gap.append({'risk':eps,'same_catalog':'FullV/BM100/BM25','oracle_cost_ms':lp.fun/N*1000,'static_random_cost_ms':ev[0]*1000,'oracle_headroom_pp':100*(ev[0]-lp.fun/N)/tt[:,0].mean(),'status':'OFFLINE_GOLD_ORACLE_NOT_DEPLOYABLE'})
idx=C.load('cache/indexes/icde_eab_independent_007_biomedical_colqwen_001/index_manifest.json','NEW_INDEX_SETUP');ocr=C.load(C.ART/'T1_preparation/ocr_001/setup_cost.json','OCR_SETUP');setup=[]
for r in main:
    base_index_wall=0 if r['method']=='BM25' else float(idx['setup_wall_s']);base_index_gpu=0 if r['method']=='BM25' else float(idx['setup_GPU_activity_s'])
    for Q in [160,1000,10000,100000,1000000]:
        wall_setup=base_index_wall+float(ocr['wall_s']);gpu_setup=base_index_gpu;setup.append({'method':r['method'],'Q':Q,'OCR_wall_s':ocr['wall_s'],'index_wall_s':base_index_wall,'index_GPU_activity_s':gpu_setup,'warm_E2E_ms':r['E2E_mean_ms'],'setup_plus_warm_ms':r['E2E_mean_ms']+1000*wall_setup/Q,'GPU_setup_plus_warm_s':float(r['GPU_active_s'])+gpu_setup/Q,'scope':'upper shared setup scenario; common OCR counted for all; FullV implementation builds BM25 unsupervised context but visual-only marginal OCR not logically necessary; model loading/BM25 build separately logged, human labeling/energy/storage monetary costs missing; not complete business net benefit'})
for name,x in [('biomedical_main_table.csv',main),('biomedical_per_query.csv',per),('biomedical_quality_cost_frontier.csv',front),('biomedical_matched_risk_analysis.csv',matched),('biomedical_group_bootstrap.csv',boots),('biomedical_oracle_gap.csv',gap),('gpu_cost_breakdown.csv',profiles),('biomedical_setup_amortization.csv',setup)]:C.table(C.R/name,x)
bad=sorted([r for r in per if r['method']=='QP_regret'],key=lambda x:-float(x['r']))[:10]
C.prose(C.R/'biomedical_failure_cases.md','# 预注册策略真实负结果\n\n'+ '\n'.join(f'- query {x["query_id"]}: action {x["action"]}, r={float(x["r"]):.6f}, BM25 Recall50={float(x["BM25_Recall50"]):.3f}; output/ref见逐查询表。' for x in bad)+'\n\n不删除失败/高伤害查询。特征固定迁移自Fit81，文档IDF仅用无标注语料；没有根据本域标签重新拟合。\n')
process_means=[T[:,p,0].mean() for p in range(3)]
C.write(C.A/'measurement_diagnostics.json',{'FullV_process_mean_s':process_means,'max_min_ratio':max(process_means)/min(process_means),'strict_streaming_AA':'NOT_RUN; no Serving advantage claimed','new_timings':len(raw),'new_GPU_profiles':len(profiles),'quality_deterministic_across_processes':True})
facts='Biomedical160全部冻结策略完成新真实路径测量，3独立进程×3重复；均值r/E2E(ms)：'+ '; '.join(f'{x["method"]}={x["mean_r"]:.5f}/{x["E2E_mean_ms"]:.2f}' for x in main)
C.prose(C.R/'T2_decision.md','# PASS_EXECUTION / DESCRIPTIVE_NOT_CERTIFIED\n\n'+facts+'\n\n主结果为冻结策略实测风险/成本；同风险包络与Oracle为回顾性描述，不是部署策略。区间按查询族和27来源双口径，预训练重叠UNKNOWN，不声称完全预训练独立。F1：物理计划前沿见主表与包络；F2：路由相对随机包络的增量见matched_risk，负值保留。\n')
C.finish('T2_summary_001','PASS_PREREGISTERED_NEW_BENCHMARK_EXECUTED',facts,'先独立上报T2再推进ColPali四域扩展；不调参、不揭封任何Cal/Eval。')
