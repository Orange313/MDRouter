import common as C
import json,collections
import numpy as np
C.A=C.ART/'X1';C.R=C.REPORT/'X1'
monitor=C.A/'resource_monitor_001';driver=C.A/'driver_002'
assert (monitor/'deadline_reached.json').exists() or (driver/'result.json').exists(),'resource monitoring still active'
result=C.load(driver/'result.json');assert result['status']=='BLOCKED_GPU_BUSY','Pilot outputs require dedicated numerical finalization; do not mislabel executed work'
polls=[json.loads(x) for x in (monitor/'polls.jsonl').read_text().splitlines()];valid=C.load(C.A/'CPU_loader_003/result.json');assert valid['checked_adapter_tensors']==254
allow=C.load(C.R/'pilot_query_allowlist.csv');old=C.load('reports/icde_eab_main_006/X2/rebuild_002/all_per_query.csv');front=[];correct=[]
for d in [C.DOM[0],C.DOM[2]]:
    ids={int(x['query_id']) for x in allow if x['domain']==d}
    for plan in ['Full-Visual','BM25-R50','BM25-R100']:
        rows=[r for r in old if r['dataset']==d and int(r['query_id']) in ids and r['policy']==plan];assert len(rows)==12
        front.append({'dataset':d,'retriever':'ColQwen2-v1.0 primary','plan':plan,'N_queries':12,'N_queries_newly_executed':0,'nDCG10':np.mean([float(r['nDCG10']) for r in rows]),'r_pos':np.mean([float(r['mean_r']) for r in rows]),'warm_E2E_ms':np.mean([float(r['E2E_s'])*1000 for r in rows]),'GPU_active_s':None,'CPU_ms':None,'index_setup_s':None,'status':'HISTORICAL_DEV_SUBSET_REUSE_NOT_NEW_PILOT','evidence':'DEV_EXPLORATORY'})
        front.append({'dataset':d,'retriever':'ColPali-v1.3 secondary','plan':plan,'N_queries':0,'N_queries_newly_executed':0,'nDCG10':None,'r_pos':None,'warm_E2E_ms':None,'GPU_active_s':None,'CPU_ms':None,'index_setup_s':None,'status':'NOT_RUN_GPU_BUSY','evidence':'NO_GPU_QUALITY_COST_EVIDENCE'})
        correct.append({'dataset':d,'plan':plan,'planned_queries':12,'actual_GPU_quality_queries':0,'CPU_adapter_tensor_checks':254,'scorer_and_top10_check':'NOT_RUN','status':'NOT_RUN_RESOURCE_BLOCKED'})
C.table(C.R/'cross_retriever_frontier.csv',front);C.table(C.R/'second_visual_correctness.csv',correct)
C.prose(C.R/'failure_cases.md','# 保留的失败与停止\n\n1. CPU_loader_001: Transformers5对旧adapter名称映射不兼容，252个语言层LoRA missing/unexpected；active_adapters存在不足以证明训练权重加载。另SizeDict写JSON失败留下不完整result.json，其中开头PASS字样不构成成功；本轮从未用该模型计算Dev质量或GPU结果。原日志/部分JSON原样保留。\n2. CPU_loader_002: 张量摘要对BFloat16调用numpy失败；保留failure，不启动GPU。\n3. CPU_loader_003: 在隔离overlay中只重命名键；254个tensor值原样保持，载入后按BF16逐一精确匹配，合成query编码有限。未修改共享.venv或缓存原权重。CPU实际约20.47秒，不能当作GPU编码/索引成本。\n4. 原CPU调度driver001因未验证加载而终止自身；没有终止其他作业。driver002只采用已核验loader，等待45分钟资源窗口仍无空卡，未启动CUDA。\n5. 因未通过资源门槛，第二模型静态pilot、索引构建、扩展及其Oracle均NOT_RUN；无失败质量查询可人为删除。\n')
facts=f'PASS 第二模型不同家族/许可/10资产审计与固定HR/Industrial各12查询；CPU兼容修复后254训练adapter张量精确匹配、合成编码通过。BLOCKED：45分钟{len(polls)}次6卡监测无空闲卡，实际GPU查询0、第二索引0；未以元数据/CPU加载充当静态pilot。'
C.write(C.A/'resource_outcome.json',{'status':'BLOCKED_GPU_BUSY','polls':len(polls),'first_time':polls[0]['time_bj'],'last_time':polls[-1]['time_bj'],'idle_observations':sum(bool(x['idle_candidates']) for x in polls),'GPU_quality_queries':0,'second_index_built':False,'foreign_jobs_terminated':0,'shared_venv_modified':False,'CPU_adapter_validation':'PASS_254_TENSORS','time_bj':C.stamp()})
C.prose(C.R/'X1_decision.md','# BLOCKED_GPU_BUSY / PASS_CPU_MODEL_AUDIT\n\n'+facts+'\n\n不扩展，不写跨检索器主张。恢复条件：连续3次空闲检查、ResourceGuard和存储余量，通过锁定24-query pilot后才决定扩展。旧Q/QP对新参考INCOMPATIBLE_NOT_RETRAINED；原005证书不迁移。\n')
C.finish('X1','BLOCKED_GPU_BUSY_PASS_TRAINED_MODEL_CPU_AUDIT',facts,'第二模型GPU质量/成本与跨检索器机制PENDING；保留CPU兼容修复及失败证据，停止扩大。')
