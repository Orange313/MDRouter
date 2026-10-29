import common as C
import collections,json,numpy as np
import sys
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
C.start('T3/expansion_summary_001');historical_groups=C.load('reports/icde_final_gate_004/S2/resident_analysis_002/all_unprofiled_requests.csv');main=[];correct=[];raw=[];prof=[];setup=[]
for d in C.DOM:
    for p in range(3):
        aa=C.ART/f'T3/expansion_{d}_p{p}_001';rr=C.REPORT/f'T3/expansion_{d}_p{p}_001';done=C.load(aa/'completed.json');assert done['status']=='PASS_FOUR_DOMAIN_STATIC_EXPANSION'
        raw+=C.load(rr/'requests.csv');correct+=C.load(rr/'correctness.csv');setup.append({'domain':d,'process':p,**C.load(aa/'setup.json')})
        g=C.load(aa/'resource_guard.json','RESOURCE_GUARD_RECEIPT');assert g.get('valid') is True and not g.get('foreign_pids') and not g.get('errors'),'RESOURCE_GUARD_INVALID'
        if p==0:prof += [{**x,'domain':d} for x in C.load(rr/'GPU_profiles.csv')]
    for plan in ['Full-Visual','BM25-R50','BM25-R100','BM25']:
        rows=[x for x in raw if x['domain']==d and x['plan']==plan];qids=sorted({x['query_id'] for x in rows});tensor=np.array([[[np.mean([float(x[k]) for x in rows if x['query_id']==q and int(x['process'])==p]) for k in ['nDCG10','r','r_bar','E2E_s','CPU_s']] for p in range(3)] for q in qids]);values=tensor.mean(1);rng=np.random.default_rng(C.SEED);family_lookup={str(x['query_id']):x['family'] for x in historical_groups if x['domain']==d};ii=F.boots(np.array([family_lookup[q] for q in qids]),2000,C.SEED);ppp=rng.integers(0,3,(2000,3));b=np.array([tensor[ix][:,pi].mean((0,1)) for ix,pi in zip(ii,ppp)])
        pp=[x for x in prof if x['domain']==d and x['plan']==plan];row={'dataset':d,'retriever':'ColPali-v1.3','plan':plan,'queries':len(qids),'processes':3,'timed_requests':len(rows),'nDCG10':values[:,0].mean(),'mean_r':values[:,1].mean(),'r_bar':values[:,2].mean(),'warm_E2E_ms':1000*values[:,3].mean(),'P50_ms':1000*np.median([float(x['E2E_s']) for x in rows]),'P95_ms':1000*np.quantile([float(x['E2E_s']) for x in rows],.95),'CPU_ms':1000*values[:,4].mean(),'r_CI_low':np.quantile(b[:,1],.025),'r_CI_high':np.quantile(b[:,1],.975),'E2E95_low_ms':1000*np.quantile(b[:,3],.025),'E2E95_high_ms':1000*np.quantile(b[:,3],.975),'status':'PASS_FOUR_DOMAIN_DEV_EXPLORATORY_NOT_INDEPENDENT_TEST','index':'cache/indexes/'+C.RUN+'_colpali_'+d+'_002'}
        for comp in ['complete_path','query_encoding','retrieval_scoring']:row[comp+'_GPU_s']=np.mean([float(x['gpu_active_union_seconds']) for x in pp if x['component']==comp]) if any(x['component']==comp for x in pp) else None
        main.append(row)
C.table(C.R/'cross_retriever_main_table.csv',main);C.table(C.R/'colpali_pilot_correctness.csv',correct);C.table(C.R/'gpu_cost_breakdown.csv',prof);C.table(C.R/'setup_cost.csv',setup)
assert all(x['status']=='PASS' for x in correct)

from scipy.optimize import linprog
import sys
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
pareto=[];oracle=[];repair=[]
for d in C.DOM:
    dd=[r for r in main if r['dataset']==d]
    for r in dd:
        dominates=[s['plan'] for s in dd if s['warm_E2E_ms']<=r['warm_E2E_ms'] and s['mean_r']<=r['mean_r'] and (s['warm_E2E_ms']<r['warm_E2E_ms'] or s['mean_r']<r['mean_r'])];pareto.append({**r,'dominated_by':dominates,'frontier':not dominates})
    qq=sorted({x['query_id'] for x in raw if x['domain']==d});plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];Y=np.array([[np.mean([float(x['r']) for x in raw if x['domain']==d and x['query_id']==q and x['plan']==p]) for p in plans] for q in qq]);T=np.array([[np.mean([float(x['E2E_s']) for x in raw if x['domain']==d and x['query_id']==q and x['plan']==p]) for p in plans] for q in qq]);n=len(qq)
    for eps in [0,.01,.02,.05,.1]:
        lp=linprog(T.ravel(),A_ub=[Y.ravel()],b_ub=[eps*n],A_eq=np.kron(np.eye(n),np.ones((1,4))),b_eq=np.ones(n),bounds=(0,1),method='highs');assert lp.success;en,_=F.envelope(Y.mean(0),T.mean(0),eps);oracle.append({'dataset':d,'retriever':'ColPali-v1.3','risk':eps,'oracle_cost_ms':lp.fun/n*1000,'static_random_cost_ms':en[0]*1000,'oracle_headroom_pp':100*(en[0]-lp.fun/n)/T[:,0].mean(),'status':'GOLD_OFFLINE_DEV_EXPLORATORY'})
    for i,q in enumerate(qq):repair.append({'dataset':d,'query_id':q,'r50':Y[i,1],'r100':Y[i,2],'BM25_r':Y[i,3],'r50_harmed':Y[i,1]>1e-10,'r100_fully_repairs_r50_harm':Y[i,1]>1e-10 and Y[i,2]<=1e-10,'r100_reduces_harm':Y[i,2]<Y[i,1]-1e-10})
C.table(C.R/'cross_retriever_pareto.csv',pareto);C.table(C.R/'cross_retriever_oracle.csv',oracle);C.table(C.R/'repair_cases.csv',repair)
C.prose(C.R/'retriever_specific_failure_cases.md','# ColPali与主模型的差异保留\n\n'+ '\n'.join(f'- {d}: R50 harmed={sum(x['r50_harmed'] for x in repair if x['dataset']==d)}, R100 fully repairs={sum(x['r100_fully_repairs_r50_harm'] for x in repair if x['dataset']==d)}; all cases retained.' for d in C.DOM)+'\n\n绝对nDCG与模型特定参考正遗憾分别报告；候选覆盖正确性见逐查询correctness。不得将旧ColQwen预测/证书拷至ColPali。新旧计时不同cohort，跨检索器延迟不是同次硬件配对；各检索器内计划为匹配查询/随机顺序。')
C.prose(C.R/'T3_decision.md','# PASS_164_DEV_CROSS_RETRIEVER_STATIC_EXECUTION\n\n四域全部合法Dev164，4静态计划，3进程×3重复/查询；每域模型特定FullV、候选覆盖、稳定tie-break均核验；无新路由训练或正式认证。所有差异与负结果保留。')
C.finish('T3_expansion_summary_001','PASS_CROSS_RETRIEVER_FOUR_DEV_STATIC',json.dumps(main,ensure_ascii=False),'写回Orange后进入T4综合与T5论文；无Serving或独立认证主张。')
