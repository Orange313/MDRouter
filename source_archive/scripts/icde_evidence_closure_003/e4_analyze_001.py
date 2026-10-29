import common as C
import numpy as np,collections,json,os,sys
C.A=C.ART/'E4';C.R=C.REPORT/'E4';C.STAGE='E4';completion=C.load(C.A/'driver_completion.json');cells=[];requests=[];profiles=[];setups=[];aa=[];comparisons=[]
for d in ['Pharmaceuticals']+C.DOM:
 base=C.A/'workers'/f'{d}_main_001';ab=C.A/'workers'/f'{d}_aa_001'
 if not (base/'cell_rows.json').exists():continue
 guard=C.load(base/'resource_guard.json');assert guard['valid'];ss=C.load(base/'cell_rows.json');rr=C.load(base/'request_rows.json');cells+=ss;requests+=rr;profiles+=C.load(base/'profile_rows.json') if (base/'profile_rows.json').exists() else [{'domain':d,'status':'NOT_RUN_PROFILER_FAILED_PRIMARY_PRESERVED','GPU_activity_s':None}];setups.append({'domain':d,**C.load(base/'setup.json')})
 if (ab/'cell_rows.json').exists():
  assert C.load(ab/'resource_guard.json')['valid'];ar=C.load(ab/'request_rows.json');ac=C.load(ab/'cell_rows.json')[0];pr=next(r for r in ss if r['method']=='Full-Visual' and r['window']==1 and r['concurrency']==1 and r['load_index']==1);rel=ac['mean_latency_s']/pr['mean_latency_s']-1;aa.append({'domain':d,'control':'independent process sameFullV inputs/QPS/window/concurrency/order','relative_mean_latency_B_minus_A':rel,'latency_A_s':pr['mean_latency_s'],'latency_B_s':ac['mean_latency_s'],'status':'PASS_POINT_EQUIVALENCE_ONLY' if abs(rel)<.1 else 'INSUFFICIENT_STREAM_NOISE','limit':'one matched process pair; no independent run CI or equivalence certificate'})
 for method in ['Q_regret','QP_regret']:
  for W in [1,8,32]:
   for con in [1,2,4]:
    for load in [0,1,2]:
     q=[r for r in rr if r['method']==method and r['window']==W and r['concurrency']==con and r['load_index']==load];b=[r for r in rr if r['method']=='RandomMix_'+method and r['window']==W and r['concurrency']==con and r['load_index']==load]
     if not q or not b:continue
     q.sort(key=lambda r:r['position']);b.sort(key=lambda r:r['position']);assert [r['query_id'] for r in q]==[r['query_id'] for r in b];g=np.array([r['family'] for r in q]);u=np.unique(g);qr=np.array([r['r'] for r in q]);br=np.array([r['r'] for r in b]);qt=np.array([r['arrival_to_completion_s'] for r in q]);bt=np.array([r['arrival_to_completion_s'] for r in b]);rng=np.random.default_rng(C.SEED);boot=[np.concatenate([np.flatnonzero(g==x) for x in rng.choice(u,len(u),replace=True)]) for _ in range(2000)];ci=np.quantile([100*(1-qt[ix].mean()/bt[ix].mean()) for ix in boot],[.025,.975]);rci=np.quantile([(qr[ix]-br[ix]).mean() for ix in boot],[.025,.975]);diff=float((qr-br).mean());saving=100*(1-qt.mean()/bt.mean());status='EXPLORATORY_MATCHED_POINT' if abs(diff)<=.005 else 'INSUFFICIENT_RISK_MATCH'
     comparisons.append({'domain':d,'method':method,'baseline':'RandomMix_'+method,'window':W,'concurrency':con,'load_index':load,'actual_policy_mean_r':qr.mean(),'actual_baseline_mean_r':br.mean(),'paired_r_difference':diff,'paired_r_diff_CI_low':rci[0],'paired_r_diff_CI_high':rci[1],'arrival_latency_saving_pct':saving,'descriptive_paired_family_CI_low_pct':ci[0],'descriptive_paired_family_CI_high_pct':ci[1],'status':status,'CI_limit':'conditional workload/family resampling; shared queue interactions and single process run prevent independent deployment significance','mean_executed_service_saving_pct':100*(1-np.mean([x['service_wall_s']+x['allocated_batch_wall_s'] for x in q])/np.mean([x['service_wall_s']+x['allocated_batch_wall_s'] for x in b]))})
C.table(C.R/'streaming_window_sweep.csv',cells);C.table(C.R/'tail_latency_goodput.csv',cells);C.table(C.R/'matched_risk_e2e.csv',comparisons);C.table(C.R/'all_stream_requests.csv',requests);C.table(C.R/'separate_GPU_replays.csv',profiles);C.table(C.R/'cold_setup.csv',setups);C.table(C.R/'stream_AA_controls.csv',aa)
probe=[]
for r in cells:
 if r['method']=='QP_regret':probe.append({k:r[k] for k in ['domain','method','window','concurrency','load_index','request_n','batch_planning_CPU_s','batch_planning_wall_s','paid_Probe_wall_s','mean_latency_s','mean_r']})
C.table(C.R/'probe_amortization.csv',probe)
facts=f'真实流式请求实测：{len(cells)}工作点、{len(requests)}未剖析请求；W1/8/32×并发1/2/4×三档QPS，含实际窗口/队列/特征/Probe复用/执行；GPU单独重放剖析{len(profiles)}组。'+json.dumps(C.clean(aa),ensure_ascii=False)
decision='全部策略仍EXPLORATORY_UNCERTIFIED；仅实际风险差<=.005的点标开发匹配，其余INSUFFICIENT。单次流式运行与配对描述区间不构成独立效果显著性；新认证仍缺独立校准与授权，不改005。是否有相对最强静态/随机收益由完整表和E5判断，不能以GPU节省替代E2E。'
C.prose(C.R/'E4_decision.md','# SYSTEM_MEASURED / EXPLORATORY_UNCERTIFIED_POLICY\n\n'+facts+'\n\n'+decision+'\n\n口径：in-process resident retrieval serving harness，无网络/跨主机序列化；cold模型/index加载单列，不将warm收益称完整业务净收益。Goodput仅延迟<=200ms的吞吐，不含认证质量保证。每格64到达请求重复已有Dev；P95为有限样本描述，窗口/队列互相依赖，不是新增独立问题。GPU活动只在独立profiler重放记录，其primary latency不使用。CPU是进程实测时间、batch规划与worker线程时间分列。A/A只一对新独立进程，若不稳定则限制系统优势结论。\n')
C.finish('PASS_STREAM_SYSTEM_MEASUREMENT' if completion['all_successful'] else 'INSUFFICIENT_PARTIAL_STREAM_FAILURES',facts,decision)
