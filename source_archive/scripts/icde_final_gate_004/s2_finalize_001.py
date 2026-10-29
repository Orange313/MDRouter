import common as C
import json,numpy as np
C.A=C.ART/'S2';C.R=C.REPORT/'S2';C.STAGE='S2';__import__('sys').addaudithook(C.protected)
actual=C.load(C.R/'resident_analysis_002/static_vs_router_matched_risk.csv');aa=C.load(C.R/'resident_analysis_002/AA_resident_process_controls.csv');false=C.load(C.R/'resident_analysis_002/false_safe_breakdown.csv');gate=C.load(C.ART/'S1/AA_gate.json');stream=[];streamstate=[]
for d in C.DOM:
 p=C.A/(d+'_stream_analysis_complete.json')
 if p.exists():z=C.load(p);streamstate.append({'domain':d,'completed_processes':z['completed_processes'],'requests':z['requests'],'status':'MEASURED' if z['completed_processes']==3 else 'INSUFFICIENT_RESOURCE_OR_PARTIAL','reason':z['invalid']});stream+=z['primary']
 else:streamstate.append({'domain':d,'completed_processes':0,'requests':0,'status':'NOT_RUN','reason':'S1 light reliability gate not passed' if gate['domains'][d]['status']!='PASS' else 'required stage incomplete'})
C.table(C.R/'streaming_status_all_domains.csv',streamstate);C.table(C.R/'streaming_results_all_domains.csv',stream)
text='# S2｜动态路由相对强静态基线的最终对照\n\n**状态：SYSTEM_MEASURED / DEV_EXPLORATORY；新方法收益INSUFFICIENT，非认证、非独立评估。**\n\n'
text+='主证据采用`resident_analysis_002/`：真实同步执行完整Q/Probe/规划/所选检索路径；4域164Dev，12独立进程、11,808次未剖析请求、1,312独立GPU profiles。先前root下的composed_*及static_vs_router_matched_risk.csv是成本拼接诊断，保留但不代替实际路径主表。首次分析001的bootstrap聚合轴错误在出表前触发异常；原始测量不变，002明确验证(B,8)形状后生成全部主表，失败记录保留。\n\n'
text+='|域|策略|mean r|实际完整路径均值ms|相对同风险静态随机增量pp|paired95%区间|常驻A/A|\n|---|---|---:|---:|---:|---|---|\n'
for r in actual:
 if r['method'] in ['BM25-R100','Q_regret','QP_regret']:text+=f'|{r["domain"]}|{r["method"]}|{float(r["mean_r"]):.5f}|{1000*float(r["actual_resident_full_path_mean_E2E_s"]):.2f}|{float(r["matched_risk_increment_pp"]):.2f}|[{float(r["paired95_increment_low_pp"]):.2f},{float(r["paired95_increment_high_pp"]):.2f}]|{r["resident_AA_status"]}|\n'
text+='\n常驻进程间AA仅Finance整体PASS；其他域均保留“测量噪声/等效性不足”标记，不以这些域的点估计或区间宣称性能优越性。已按查询族+进程配对bootstrap2000次，区间逐点；不能以大量计时当独立查询。即使Finance通过，Q/QP同风险增量区间也跨零。\n\n'
text+='**同实际风险 / 同成本：** 主表同时给matched-risk随机静态包络、matched-cost最小实际r、与BM100的风险/成本差；查询无关静态随机混合的同步单请求期望成本和风险线性，因此此包络是合法描述性强对照，但其Dev/重采样重优化不构成部署策略证书。Mean nDCG–成本与positive-r–成本前沿分开，不能用signed improvement抵消正遗憾。\n\n'
text+='**Oracle gap：** `resident_analysis_002/oracle_gap_decomposition.csv`分别给准确3动作(V/R100/BM25)和4动作(+R50)Gold LP上限；只用于信息界，不是可部署方法。所有新增编码、Probe和CPU规划在实际路径中支付；QP已付BM search仅在BM-selected路径复用，转FullV额外支付。CPU/GPU活动秒分别计量，不相加当延迟。\n\n'
text+='**False-Safe：**\n\n'
for r in false:text+=f'- {r["domain"]}/{r["method"]}：r>.05的非参考选择{r["false_safe_n"]}/{r["Dev_n"]}，贡献正遗憾总和的{100*float(r["false_safe_r_share"] or 0):.1f}%；queryID与原始逐请求记录均可追踪。\n'
text+='\n**流式对照：**\n\n'
for s in streamstate:text+=f'- {s["domain"]}：{s["status"]}，{s["completed_processes"]}独立进程、{s["requests"]}条有效请求。\n'
text+='\n流式主张仅限S1通过的域与预定2/5QPS、W1/C1，所有吞吐是固定到达率下完成率，不是最大容量；Goodput只按200ms延迟，不是质量认证。流式静态latency的LP插值仅诊断，因为队列可非线性；实际frozen随机路由比较须同时报告实现风险差及其是否≤.005。未匹配风险时不得拿更冒险的低延迟当收益。资源中止的局部格保留，缺少3进程时一律INSUFFICIENT而不选有效局部格作优越性主张。\n\n'
text+='**摊销：** `setup_amortization.csv`保留Q100…1e6场景；resident+setup/Q是成本归因而非响应延迟。新索引/模型setup GPU、网络、完整主机能耗等missing继续保留，未批准部署Q，不报告完整业务净收益。\n\n'
text+='**结论—决策—阻塞：** 动态路由未稳定胜过最强静态/随机同风险对照，停止开发更复杂路由模块；Regular新方法收益尚未证明。EAB可聚焦计划目录、信息上限、风险保证与可靠测量的差距。统计功效、测量等效性、独立终评仍缺证据。\n\n**论文主张**：四域真实路径成本/两个分开的质量风险前沿/强基线控制/False-Safe主导损失/Oracle信息空间；不支持已认证或完整上线净收益。\n'
C.prose(C.R/'S2_decision.md',text);C.prose(C.R/'final_table_index.md','# 最终表路径\n\n- 实际路径主表：resident_analysis_002/static_vs_router_matched_risk.csv\n- 两类前沿：resident_analysis_002/cost_risk_frontier.csv\n- Oracle：resident_analysis_002/oracle_gap_decomposition.csv\n- False-Safe：resident_analysis_002/false_safe_breakdown.csv\n- 新GPU/CPU：resident_analysis_002/full_path_cost_decomposition.csv、GPU_profile_summary.csv\n- 流式：streaming_results_all_domains.csv、各域_stream_matched_risk.csv（root中的HR前缀保留历史脚本版本命名）\n- root中composed_*与较早static_vs_router*是成本拼接中间诊断，不覆盖。\n')
facts='PASS 实际完整路径11,808次计时、1,312GPU profiles；常驻AA仅Finance整体通过。Q/QP四域均未证明相对同风险强静态随机包络的稳定增量；PharmaQP点增量9.30pp，paired95%[−9.42,20.52]，且常驻AA不足，不能作方法收益。流式实际状态见各域表，失败/未运行均保留。'
C.finish('SYSTEM_MEASURED_METHOD_ADVANTAGE_INSUFFICIENT',facts,'动态路由没有稳定超越强静态/随机对照，不再开发复杂路由器；S3只诊断既有学习曲线，S4建议Regular NO_GO、EAB分析路线待证据和复现验收。')
