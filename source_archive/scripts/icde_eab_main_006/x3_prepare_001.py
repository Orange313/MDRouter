import common as C
import sys,json,subprocess,shutil,collections
import numpy as np
from x3_mock_pipeline_001 import run
C.start('X3')
old=C.load('artifacts/icde_eab_validation_005/A3/one_shot_eval_protocol.json','SEALED_SCOPE_METADATA_ONLY')
mock=run(C.A/'mock_full_001');denial=run(C.A/'must_not_exist','real')
assert denial['exit_code']==77 and not (C.A/'must_not_exist').exists()
C.write(C.A/'denial_smoke.json',denial)
protocol={**old,'status':'FROZEN_MOCK_AND_ANALYSIS_SPEC_REAL_ACQUISITION_INTEGRATION_INCOMPLETE','run_id':C.RUN,'time_bj':C.stamp(),'no_authorization':True,'new_methods':['RR-style adapted hard','RR-style adapted soft'],'adapted_model_source':'existing Pharma Fit81 primary seed202610271130 kappa20, no new fit or Dev selection','new_eval73_reads':0,'new_cal81_reads':0,'procedure':['verify separately explicit future Eval-only authorization and protocol/assets','provider returns exactly locked Eval73 query/qrel rows, never mixed Cal payload','three idle checks + ResourceGuard; execute frozen full paths 3 processes x 3 repeats with independent profiles','fixed primary Q-vs-BM100; secondary QP-vs-Q/mixes and adapted controls; all prespecified metrics and pointwise family intervals','publish once even if negative; no tuning/replacement/repeat after outcomes'],'real_hardware_provider':'NOT_RUN_NOT_FULLY_INTEGRATED; synthetic mock is not execution readiness proof'}
C.prose(C.R/'one_shot_eval_protocol.yaml',json.dumps(protocol,ensure_ascii=False,indent=2));C.write(C.A/'one_shot_eval_protocol.json',protocol)
C.prose(C.R/'eval_access_audit.md','# PASS sealed-access refusal / PASS full synthetic mock / INSUFFICIENT real integration\n\nSynthetic mock runs role projection → 73 mock queries ×10 methods → per-query outcomes → grouped paired summaries; 730 synthetic rows are not scientific samples. Nonmock invocation refuses before provider calls, real payload reads0. Actual Eval73/Cal81 text, qrels and outcomes not read. Mixed-role official Pharma parquet not opened. Real role-filtered acquisition/hardware provider has not been integrated or tested; do not call this one-click real Eval ready. Existing frozen source/model/index assets remain pinned by X0.\n')
C.prose(C.R/'eval73_one_shot_ready.md','# INSUFFICIENT｜尚不可实际揭封\n\nPASS: protocol/schema, synthetic role isolation, 730 mock policy rows, paired analysis, default-deny before provider. NOT_RUN: official Eval-only acquisition projection, real retrieval integration/hardware smoke and source hash freeze for those adapters. No real Cal/Eval data read. Must finish nonsealed provider mock plus physical Dev adapter smoke before separate user release.\n')
C.prose(C.R/'explicit_authorization_request.md','# 尚未满足请求揭封的全部条件\n\n本轮不请求立即批准。日后只可单独申请Pharma Eval73的一次性描述性评测，不含Cal81/B-L/test/CS。先补齐并冻结真实获取与硬件接口、核验模型与命令hash、完整Dev替身空跑；之后再由用户明确授权。结果只支持固定语料query benchmark描述，不是新K1、未见文档泛化或生产IID。\n')
raw=C.load('reports/icde_eab_validation_005/A2/new_valid_requests.csv');cells=C.load('reports/icde_eab_validation_005/A2/aa_cells_v3.csv');decomp=[]
for d in [C.DOM[0],C.DOM[3]]:
    for qps in [2,5]:
        for block in [0,1]:
            for a,b in [('A1','B1'),('A2','B2')]:
                z={s:next(x for x in cells if x['domain']==d and float(x['QPS_offered'])==qps and int(x['block'])==block and x['slot']==s) for s in [a,b]}
                av,bv=[float(z[s]['mean_arrival_to_completion_s']) for s in [a,b]];sa,sb=[float(z[s]['mean_service_wall_s']) for s in [a,b]]
                decomp.append({'domain':d,'QPS':qps,'block':block,'A':a,'B':b,'service_log_ratio':np.log(sb/sa),'total_log_ratio':np.log(bv/av),'queue_dispatch_log_contribution':np.log(bv/av)-np.log(sb/sa),'A_queue_dispatch_ms':1000*(av-sa),'B_queue_dispatch_ms':1000*(bv-sb),'accounting_identity':'total=dispatch_lateness+executor_queue+service'})
C.table(C.R/'queue_vs_service_pair_decomposition.csv',decomp)
hr=[x for x in decomp if x['domain']==C.DOM[0] and x['QPS']==5]
C.prose(C.R/'serving_AA_gate.md','# INSUFFICIENT｜AA严格门槛保持\n\n原2720条新AA请求和32 cells重析，16个配对对照。HR5QPS服务时间98.75%CI[.95763,1.01778]在等效区间内，总响应[.87527,1.08719]下端越界。逐进程配对log(total_B/total_A)减log(service_B/service_A)给出队列与dispatch相对贡献：'+json.dumps(hr,ensure_ascii=False)+'\n\n该分解是恒等式而非因果识别；少量队列/dispatch的进程异质性也能扩大对数比区间，4对进程低功效。不能断言排队是唯一原因，也不能删慢请求或选幸运QPS。新GPU AA尚未运行；全部GPU繁忙时继续BLOCKED_RESOURCE，跨策略Serving NOT_RUN。原98.75%CI全落[.9,1.1]且2/5QPS两指标全过才允许比较，不以HR服务时间单项过替代。\n')
C.write(C.A/'prepared.json',{'time_bj':C.stamp(),'mock_status':mock['status'],'mock_rows':730,'denial':'PASS','real_eval_reads':0,'queue_pairs':len(decomp),'external_audit':'PENDING_SOURCE_ACQUISITION','new_AA':'NOT_RUN_RESOURCE_PENDING'})
print(json.dumps({'status':'X3_MOCK_AND_QUEUE_ANALYSIS_PREPARED','time_bj':C.stamp()}),flush=True)
