"""CPU main-table rebuild. No fitting; frozen same-capacity hard/soft reuse."""
import common as C
import sys, collections, json, time
import numpy as np
from scipy.optimize import linprog
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
import learning_base_001 as L
C.start('X2/rebuild_002')
raw=C.load('reports/icde_final_gate_004/S2/resident_analysis_002/all_unprofiled_requests.csv')
gpu=C.load('reports/icde_final_gate_004/S2/resident_analysis_002/GPU_profile_summary.csv')
oldmaster=C.load('reports/icde_eab_validation_005/A1/matched_risk_cost_master.csv')
models=C.load('artifacts/icde_evidence_closure_003/E2/candidate_models.json','FROZEN_FIT81_MODELS')
locked=C.load(C.ART/'X0/locked_models.json')
costspec=C.load(C.ART/'X0/domain_policy_config.json')
oldmethods=C.PLANS+['Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret']
catalog=[0,2,3];B=2000
main=[];pq=[];front=[];gaps=[];predaudit=[];timing=[];curves=[]
def unpack(m):
    a=[np.array(x) for x in m['model']];a[0]=a[0].astype(bool);return tuple(a)
def ci(z):return np.quantile(z,[.025,.975]).tolist()
for di,d in enumerate(C.DOM):
    rr=[r for r in raw if r['domain']==d];ids=sorted({int(r['query_id']) for r in rr});N=len(ids)
    groups=np.array([next(r['family'] for r in rr if int(r['query_id'])==q) for q in ids])
    grouped=collections.defaultdict(list)
    for r in rr:grouped[int(r['query_id']),int(r['process']),r['method']].append(r)
    keys=['resident_full_path_E2E_s','r','nDCG10','CPU_thread_s','Q_wall_s','Probe_search_extra_wall_s','Pstats_extra_wall_s','prediction_decision_wall_s']
    arr={k:np.array([[[np.mean([float(r[k]) for r in grouped[q,p,m]]) for m in oldmethods] for p in range(3)] for q in ids]) for k in keys}
    T,Y,U,CPU=[arr[k] for k in keys[:4]]
    if di<3:
        a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz');ix=[list(a['qids']).index(q) for q in ids];Q=a['Q'][ix];P=a['P'][ix]
    else:
        a=C.load('artifacts/icde_evidence_closure_003/E2/pharma_dataset.npz');ix=[list(a['qids']).index(q) for q in ids];Q=a['Q'][ix];P=a['P'][ix]
    aa=next(r['resident_AA'] for r in oldmaster if r['domain']==d)
    # Check that reused query feature semantics agree with measured frozen Q decisions.
    qm=next(x for x in locked if x['method']=='Q_regret');pr=L.predict(unpack(qm),Q)
    cvec=np.array(next(x for x in costspec if x['domain']==d)['label_free_cost_vector'])
    acts=np.argmin(cvec[catalog][None]+20*np.maximum(pr[:,catalog],0),axis=1)
    for i,q in enumerate(ids):
        actual={r['action'] for p in range(3) for r in grouped[q,p,'Q_regret']}
        predaudit.append({'domain':d,'query_id':q,'reused_feature_prediction':C.PLANS[catalog[acts[i]]],'actual_frozen_actions':sorted(actual),'status':'PASS' if actual=={C.PLANS[catalog[acts[i]]]} else 'INCOMPATIBLE_FEATURE_CONTEXT'})
    compatible=all(r['status']=='PASS' for r in predaudit if r['domain']==d)
    extra=[]
    for target in ['hard','soft']:
        name='RR-style adapted '+target
        model=next(x for x in models if x['source']=='Pharma_only' and x['seed']==202610271130 and x['method']=='Q_'+target and x['kappa']==20)
        m=unpack(model)
        # No retraining, no Dev grid search; fixed inherited kappa=20. BM50 is masked identically to Q/QP.
        pp=L.predict(m,Q);action=np.array(catalog)[np.argmax(pp[:,catalog],axis=1)]
        infer_wall=[];infer_cpu=[]
        for i,q in enumerate(ids):
            w=[];c=[]
            for rep in range(3):
                st=time.perf_counter();ct=time.process_time();pr=L.predict(m,Q[i:i+1]);_a=int(np.array(catalog)[np.argmax(pr[:,catalog])]);c.append(time.process_time()-ct);w.append(time.perf_counter()-st)
                assert _a==action[i]
            infer_wall.append(np.mean(w));infer_cpu.append(np.mean(c));timing.append({'domain':d,'query_id':q,'method':name,'CPU_inference_repeats':3,'wall_s':w,'CPU_s':c,'retrieval_executed':False})
        iw=np.array(infer_wall);ic=np.array(infer_cpu);qwall=arr['Q_wall_s'][:,:,4].mean(1)
        et=np.stack([T[i,:,action[i]]+qwall[i]+iw[i] for i in range(N)])
        ey=np.stack([Y[i,:,action[i]] for i in range(N)]);eu=np.stack([U[i,:,action[i]] for i in range(N)])
        ec=np.stack([CPU[i,:,action[i]]+ic[i] for i in range(N)])
        extra.append((name,et,ey,eu,ec,action,iw,compatible))
    TT=np.concatenate([T]+[x[1][:,:,None] for x in extra],axis=2)
    YY=np.concatenate([Y]+[x[2][:,:,None] for x in extra],axis=2)
    UU=np.concatenate([U]+[x[3][:,:,None] for x in extra],axis=2)
    CC=np.concatenate([CPU]+[x[4][:,:,None] for x in extra],axis=2)
    names=oldmethods+[x[0] for x in extra]
    idx=F.boots(groups,B,C.SEED+di);rng=np.random.default_rng(C.SEED+di);proc=rng.integers(0,3,(B,3))
    bt=np.array([TT[i][:,p].mean((0,1)) for i,p in zip(idx,proc)]);by=np.array([YY[i][:,p].mean((0,1)) for i,p in zip(idx,proc)])
    for mi,name in enumerate(names):
        meanT=TT[:,:,mi].mean();meanR=YY[:,:,mi].mean();meanU=UU[:,:,mi].mean()
        ev,weights=F.envelope(YY[:,:,:4].mean((0,1)),TT[:,:,:4].mean((0,1)),meanR)
        eb,_=F.envelope(by[:,:4],bt[:,:4],by[:,mi]);inc=100*(eb-bt[:,mi])/bt[:,0];lo,hi=ci(inc)
        eg,ew=F.envelope(YY[:,:,catalog].mean((0,1)),TT[:,:,catalog].mean((0,1)),meanR)
        gcost=[float(r['gpu_active_union_seconds']) for r in gpu if r['domain']==d and r['method']==name] if mi<8 else [float(r['gpu_active_union_seconds']) for i,q in enumerate(ids) for r in gpu if r['domain']==d and int(r['query_id'])==q and r['method']==C.PLANS[extra[mi-8][5][i]]]
        originals=[r for r in rr if r['method']==name]
        direct=mi<8;status='PASS_REBUILT_DEV' if direct else ('EXPLORATORY_COMPOSITE_COST' if compatible else 'INCOMPATIBLE_NOT_EVALUATED')
        # Do not invent new total-path quantiles or full CPU for recomposed baselines.
        row={'dataset':d,'retriever':'ColQwen2-v1.0','action_policy':name,'N_queries':N,'N_families':len(set(groups)),'source':'ViDoRe V3','split':'Dev','nDCG@10':meanU if direct or compatible else None,'r_pos_mean':meanR if direct or compatible else None,'harm_rate':(YY[:,:,mi].mean(1)>1e-10).mean() if direct or compatible else None,'wall_E2E_mean_ms':meanT*1000 if direct or compatible else None,'wall_P95_ms':np.quantile([float(r['resident_full_path_E2E_s'])*1000 for r in originals],.95) if direct else None,'GPU_active_s':np.mean(gcost) if gcost and (direct or compatible) else None,'GPU_scope':'historical separate instrumented cohort' if direct else 'composed selected static historical GPU profiles; not new GPU run','CPU_ms':CC[:,:,mi].mean()*1000 if direct else None,'CPU_known_partial_ms':None if direct else CC[:,:,mi].mean()*1000,'probe_overhead_ms':1000*(arr['Probe_search_extra_wall_s'][:,:,mi].mean()+arr['Pstats_extra_wall_s'][:,:,mi].mean()) if direct else 0.0,'query_feature_ms':arr['Q_wall_s'][:,:,mi].mean()*1000 if direct else arr['Q_wall_s'][:,:,4].mean()*1000,'planner_ms':arr['prediction_decision_wall_s'][:,:,mi].mean()*1000 if direct else np.mean(extra[mi-8][6])*1000,'index_setup_s':None,'amortization_assumptions':'warm resident; index shared/reused; setup Q separate, not user fixed','risk_matched_cost_gain_pp':100*(ev[0]-meanT)/TT[:,:,0].mean() if direct or compatible else None,'paired_family_CI_low_pp':lo if direct or compatible else None,'paired_family_CI_high_pp':hi if direct or compatible else None,'same_catalog_3arm_envelope_ms':eg[0]*1000,'full_4static_envelope_ms':ev[0]*1000,'A/A_gate':aa,'evidence_level':'DEV_EXPLORATORY','status':status,'cost_scope':'historical actual complete unprofiled path, 3 processes x 3 repeats/query' if direct else 'selected static measured path + recorded query features + fresh CPU inference; NOT freshly measured complete route','baseline_fidelity':'ADAPTED_NOT_FAITHFUL' if mi>=8 else 'LOCAL_FIXED_IMPLEMENTATION','policy_actions':'FullV/BM100/BM25' if mi in [4,5,8,9] else 'see allocation ledger'}
        main.append(row)
        for i,q in enumerate(ids):pq.append({'dataset':d,'query_id':q,'family':groups[i],'policy':name,'mean_r':YY[i,:,mi].mean(),'nDCG10':UU[i,:,mi].mean(),'E2E_s':TT[i,:,mi].mean(),'cost_scope':row['cost_scope'],'action':C.PLANS[extra[mi-8][5][i]] if mi>=8 else json.dumps(collections.Counter(r['action'] for p in range(3) for r in grouped[q,p,name])),'status':status})
    # Same-risk random envelopes: expectations, not deployed randomized latency samples.
    for eps in [.02,.05,.10]:
        ev,w=F.envelope(YY[:,:,:4].mean((0,1)),TT[:,:,:4].mean((0,1)),eps)
        eb,_=F.envelope(by[:,:4],bt[:,:4],np.full(B,eps));gain=100*(1-eb/bt[:,0]);lo,hi=ci(gain)
        front.append({'dataset':d,'epsilon_descriptive':eps,'catalog':'FullV/R50/R100/BM25','mean_cost_ms':1000*ev[0],'allocation':w[0].tolist(),'saving_vs_FullV_pp':100*(1-ev[0]/TT[:,:,0].mean()),'paired95_low_pp':lo,'paired95_high_pp':hi,'cost_scope':'IDEAL_FRACTIONAL_EXPECTATION_NO_ALLOCATION_OVERHEAD','evidence_level':'DEV_EXPLORATORY','certificate':'NOT_CERTIFIED'})
    for eps in [0,.01,.02,.05,.10]:
        yt=YY[:,:,:4].mean(1)[:,catalog];tt=TT[:,:,:4].mean(1)[:,catalog]
        lp=linprog(tt.ravel(),A_ub=[yt.ravel()],b_ub=[eps*N],A_eq=np.kron(np.eye(N),np.ones((1,3))),b_eq=np.ones(N),bounds=(0,1),method='highs');assert lp.success
        ec=F.envelope(yt.mean(0),tt.mean(0),eps)[0][0]
        gaps.append({'dataset':d,'risk_budget':eps,'catalog':'FullV/BM100/BM25','oracle_cost_ms':lp.fun/N*1000,'static_random_cost_ms':ec*1000,'oracle_headroom_pp':100*(ec-lp.fun/N)/tt[:,0].mean(),'oracle_r':float(lp.x@yt.ravel()/N),'status':'OFFLINE_ORACLE_NOT_DEPLOYABLE'})
    print(json.dumps({'stage':'X2','domain':d,'queries':N,'RR_feature_context_compatible':compatible,'time_bj':C.stamp()}),flush=True)
C.table(C.R/'eab_main_table_dev.csv',main);C.table(C.R/'risk_matched_frontier.csv',front);C.table(C.R/'oracle_learnability_gap.csv',gaps);C.table(C.R/'all_per_query.csv',pq);C.table(C.R/'feature_compatibility_audit.csv',predaudit);C.table(C.R/'CPU_router_inference_timings.csv',timing)
C.prose(C.R/'baseline_fidelity_matrix.md','# 基线忠实度与公平性\n\nOriginal RetrievalRouter reproduction: NOT_RUN。官方仓库commit 6fca6dd498161d0a49820b1b0b45060b1e976306的arms.py为MM-rerank/MM-single/Text-rerank/Text-single/BM25五动作；Full late interaction只作参考。Qwen3-0.6B-Base + LoRA r16、alpha32、dropout.05、KL软效用目标、temperature.1、nDCG@5和normalized latency。官方固定15ms路由延迟不能搬来当本地实测。\n\n本地RR-style adapted hard/soft：复用已有Pharma Fit81、seed202610271130、kappa20模型，无新拟合或搜索；150维query + 相同Ridge容量，L2来自原Fit内部grouped tuning。动作同Q/QP限制FullV/BM100/BM25；soft targets=softmax(-(normalized cost+20*r)/.1)，Ridge平方损失拟合而非KL，目标r为nDCG@10正遗憾。因此为ADAPTED_NOT_FAITHFUL，只是同容量信息/监督对照，不能称优于原论文。旧三域为固定Pharma模型迁移，全部评价是曝光过的Dev。\n\n8个既有方法真实完整路径、3进程x3重复；两个adapted策略只做CPU推理与已测静态路径组合，P95及完整CPU留missing，不冒充新硬件全路径。主表明确cost_scope。RandomMix实测路径与零分配开销理想静态随机包络分开。四静态catalog包络是更强对照；三个共同动作catalog也另列。所有区间为2000次配对family+process bootstrap，逐点、条件性，不是IID或认证。\n')
facts=f'实际CPU重建{len(main)}行四域主表，输入11808条历史完整路径+1312独立GPU剖析；164 Dev/122候选族。复用Fit81冻结hard/soft，新增{len(timing)}条CPU推理测量；完整RR路径和P95未新测，明确组合成本/缺失。动态路由增量未获独立证据。'
C.prose(C.R/'X2_decision.md','# PASS_CPU_REBUILD / INSUFFICIENT_FAITHFUL_BASELINE_AND_INDEPENDENT_GAIN\n\n'+facts+'\n\n不得将40行表都称新实测。8个基线x4=32行来自原硬件请求，8行adapted只为CPU与组合估计。第二模型和独立评估另门槛；停止复杂模型。\n')
C.finish('X2_rebuild_002','PASS_CPU_REBUILD_INSUFFICIENT_FULL_RR_PATH',facts,'先上报X2；支持EAB公平对照与信息开销分析，不支持新SOTA/正式认证/Serving优越性。')
