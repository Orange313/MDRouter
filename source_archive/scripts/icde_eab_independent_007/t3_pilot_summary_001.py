import common as C
import collections,json,numpy as np
C.start('T3/pilot_summary_001');main=[];correct=[];raw=[];prof=[];setup=[]
for d in ['vidore_v3_hr','vidore_v3_industrial']:
    for p in range(3):
        aa=C.ART/f'T3/pilot_{d}_p{p}_002';rr=C.REPORT/f'T3/pilot_{d}_p{p}_002';done=C.load(aa/'completed.json');assert done['status']=='PASS_SMALL_STATIC_PILOT'
        raw+=C.load(rr/'requests.csv');correct+=C.load(rr/'correctness.csv');setup.append({'domain':d,'process':p,**C.load(aa/'setup.json')})
        g=C.load(aa/'resource_guard.json','RESOURCE_GUARD_RECEIPT');assert g.get('valid') is True and not g.get('foreign_pids') and not g.get('errors'),'RESOURCE_GUARD_INVALID'
        if p==0:prof += [{**x,'domain':d} for x in C.load(rr/'GPU_profiles.csv')]
    for plan in ['Full-Visual','BM25-R50','BM25-R100']:
        rows=[x for x in raw if x['domain']==d and x['plan']==plan];qids=sorted({x['query_id'] for x in rows});tensor=np.array([[[np.mean([float(x[k]) for x in rows if x['query_id']==q and int(x['process'])==p]) for k in ['nDCG10','r','r_bar','E2E_s','CPU_s']] for p in range(3)] for q in qids]);values=tensor.mean(1);rng=np.random.default_rng(C.SEED);ii=rng.integers(0,len(qids),(2000,len(qids)));ppp=rng.integers(0,3,(2000,3));b=np.array([tensor[ix][:,pi].mean((0,1)) for ix,pi in zip(ii,ppp)])
        pp=[x for x in prof if x['domain']==d and x['plan']==plan];row={'dataset':d,'retriever':'ColPali-v1.3','plan':plan,'queries':len(qids),'processes':3,'timed_requests':len(rows),'nDCG10':values[:,0].mean(),'mean_r':values[:,1].mean(),'r_bar':values[:,2].mean(),'warm_E2E_ms':1000*values[:,3].mean(),'P50_ms':1000*np.median([float(x['E2E_s']) for x in rows]),'P95_ms':1000*np.quantile([float(x['E2E_s']) for x in rows],.95),'CPU_ms':1000*values[:,4].mean(),'r_CI_low':np.quantile(b[:,1],.025),'r_CI_high':np.quantile(b[:,1],.975),'E2E95_low_ms':1000*np.quantile(b[:,3],.025),'E2E95_high_ms':1000*np.quantile(b[:,3],.975),'status':'PASS_PILOT_ONLY_DEV12_NOT_INDEPENDENT_TEST','index':'cache/indexes/'+C.RUN+'_colpali_'+d+'_002'}
        for comp in ['complete_path','query_encoding','retrieval_scoring']:row[comp+'_GPU_s']=np.mean([float(x['gpu_active_union_seconds']) for x in pp if x['component']==comp])
        main.append(row)
C.table(C.R/'cross_retriever_main_table.csv',main);C.table(C.R/'colpali_pilot_correctness.csv',correct);C.table(C.R/'gpu_cost_breakdown.csv',prof);C.table(C.R/'setup_cost.csv',setup)
assert all(x['status']=='PASS' for x in correct)
C.prose(C.R/'T3_decision.md','# PASS_24_QUERY_PILOT\n\nHR12+Industrial12，3进程×3重复每计划，正确性全部PASS；先完成Biomedical160最高优先级，再判断四域扩展预算。两次001配置绑定失败在质量计算前，失败日志完整保留；002仅修复BM25 CFG依赖，未调整任何科学参数。\n\n各自模型FullV为风险参考，不搬用旧Q/QP预测，不称独立评估或正式认证。index GPU setup未测保留missing；每请求编码/检索独立profile，不将wall当GPU。\n')
C.finish('T3_pilot_summary_001','PASS_COLPALI_24_QUERY_ACTUAL_GPU_PILOT',json.dumps(main,ensure_ascii=False),'先独立上报pilot，再按顺序运行Biomedical；扩展不能挤占新域主实验。')
