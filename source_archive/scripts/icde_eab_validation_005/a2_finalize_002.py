import common as C,json,numpy as np,pathlib,shutil
C.A=C.ART/'A2';C.R=C.REPORT/'A2';C.STAGE='A2';sys=__import__('sys');sys.addaudithook(C.protected);done=C.load(C.A/'combined_complete_004.json');schedule=done['schedule'];gates=done['AA_checks'].copy();rows=[];cells=[];partial=[];tele=[]
for s in schedule:
 if 'path' not in s:continue
 p=pathlib.Path(s['path']);valid=s['status']=='PASS_WORKER_COMPLETE'
 if valid:
  rows+=C.load(p/'request_rows.json');cells+=C.load(p/'cell_rows.json')
 elif (p/'all_completed_invokes.json').exists():partial+=C.load(p/'all_completed_invokes.json')
 if (p/'telemetry.json').exists():tele+=C.load(p/'telemetry.json')
for d in C.DOM:
 if d not in [x['domain'] for x in gates]:gates.append(dict(domain=d,QPS=None,metric=None,ratio=None,CI_low=None,CI_high=None,status='NOT_RUN_RESOURCE_BLOCKED' if d in [C.DOM[0],C.DOM[3]] else 'NOT_RUN_NO_DOMAIN_SPECIFIC_AA_PASS',paired_process_n=0,CI_level=.9875))
C.table(C.R/'aa_equivalence_v3_all_domains.csv',gates);C.table(C.R/'new_valid_requests.csv',rows,fields=list(rows[0]) if rows else ['domain','method','query_id','status']);C.table(C.R/'new_invalid_partial_requests.csv',partial,fields=list(partial[0]) if partial else ['domain','query_id','status']);C.table(C.R/'aa_cells_v3.csv',[x for x in cells if x['phase']=='light'],fields=list(cells[0]) if cells else ['domain','method','status']);policycells=[x for x in cells if x['phase']=='routing'];C.table(C.R/'stable_load_policy_comparison.csv',policycells,fields=list(policycells[0]) if policycells else ['domain','method','QPS_offered','requests','mean_service_wall_s','mean_arrival_to_completion_s','P50_s','P95_s','goodput_QPS_le200ms','completed_QPS','process_CPU_s','GPU_activity_s','status'])
for n in ['old_queue_service_breakdown.csv','old_raw_AA_requests.csv','prior004_light_load_summary.csv','failure_regime.md','old_queue_amplification.png','old_queue_amplification.pdf']:shutil.copyfile(C.R/'trace_diagnostics_001'/n,C.R/n)
if not rows:
 C.table(C.R/'streaming_not_run.csv',[dict(domain=d,AA='NOT_RUN_NO_IDLE_GPU',static_or_policy_comparison='NOT_RUN',P50=None,P95=None,Goodput=None,throughput=None,actual_request_GPU_cost=None,reason='No eligible continuous idle GPU in predeclared wait window; no fabricated zeros') for d in [C.DOM[0],C.DOM[3]]])
mon=[json.loads(s) for s in (C.A/'resource_monitor.jsonl').read_text().splitlines()];C.write(C.A/'resource_summary.json',{'first_poll_bj':mon[0]['time_bj'],'last_poll_bj':mon[-1]['time_bj'],'poll_n':len(mon),'last_snapshot':mon[-1]['GPU_rows'],'new_valid_requests':len(rows),'invalid_partial_requests':len(partial),'no_idle_selected_cards_all_polls':all(all(mem>=128 or u!=0 for g,mem,u in x['GPU_rows'] if g in [0,1,4]) for x in mon),'predeclared_allowed_GPU_subset':[0,1,4],'other_GPU_observed_but_not_added_posthoc':[2,3,5]})
qualified=[d for d in C.DOM if any(x['domain']==d for x in gates) and all(x['status']=='PASS' for x in gates if x['domain']==d)];facts=f'PASS 旧原始512条A/A与004轻负载5248条重析；本轮新有效请求{len(rows)}、污染/中止部分{len(partial)}。已固定HR/Pharma2/5QPS ABBA、暖机和门槛，但资源监测{len(mon)}次，未获符合预登记条件的空闲卡时不启动CUDA。本轮全门槛通过域={qualified}，策略cell={len(policycells)}；未通过域停止策略比较，不重跑争取PASS。实际P50/P95/吞吐与board能耗见表；新kernel-activeGPU秒missing，不能填0或套用旧profile。不作未支持的Serving优越性主张。'
status='BLOCKED_GPU_NEW_SERVING_NOT_RUN' if not rows else ('INSUFFICIENT_AA_RELIABILITY' if len(qualified)<2 else 'MEASURED_DOMAIN_SPECIFIC_EXPLORATORY_NO_FORMAL_GUARANTEE')
C.prose(C.R/'A2_decision.md','# '+status+'\n\n'+facts+'\n\n原有其他进程不终止。CPU trace诊断完成不能替代新系统实验；没有请求时资源成本、分位数与置信区间为missing。Finance/Industrial没有本轮相同协议A/A，因此不作优越性主张。预CUDA调度器的同GPU绑定修订与失败命令均保留，发生在任何新GPU请求前，未按结果选择幸运运行。未修改旧protocol005/数据/假设。\n')
C.finish(status,facts,'Serving证据受资源阻塞，保留已完成CPU结果；EAB仍非投稿就绪。以后有连续空闲资源需新attempt，保持预先固定的统计门槛，不复用旧PASS冒充新测量。')
