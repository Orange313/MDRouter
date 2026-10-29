import common as C
import json,numpy as np,pathlib
C.A=C.ART/'S1';C.R=C.REPORT/'S1';C.STAGE='S1';__import__('sys').addaudithook(C.protected);gate=C.load(C.A/'AA_gate.json');rows=C.load(C.R/'AA_equivalence_gate.csv');requests=C.load(C.R/'aa_per_request.csv');tele=C.load(C.R/'CPU_GPU_telemetry_summary.csv');old=C.load(C.R/'queue_service_decomposition.csv');stress=[];failures=[]
for p in (C.A/'workers').rglob('failure.json'):failures.append({'path':str(p.relative_to(C.ROOT)),**json.loads(p.read_text())})
for p in (C.A/'workers').glob('*_stress_*/cell_rows.json'):
 g=C.load(p.parent/'resource_guard.json');z=C.load(p);stress.extend([{**r,'guard_valid':g['valid']} for r in z])
C.table(C.R/'medium_near_load_diagnostics.csv',stress);C.write(C.A/'failure_inventory.json',failures)
primary_n=sum(r['primary_scheduled']=='True' for r in requests);extra_n=len(requests)-primary_n;passed=[d for d,v in gate['domains'].items() if v['status']=='PASS'];text='# S1｜流式测量信度\n\n'
text+='**结论**：旧E4跨策略性能主张仍无效。新轻负载预登记A/A按四个校正区间全部通过才开放该域流式比较，未降低阈值、未换seed。\n\n|领域|QPS|指标|B/A|校正98.75%区间|统计状态|\n|---|---:|---|---:|---|---|\n'
for r in rows:text+=f'|{r["domain"]}|{float(r["QPS"]):g}|{r["metric"]}|{float(r["B_A_ratio"]):.4f}|[{float(r["CI_low"]):.4f},{float(r["CI_high"]):.4f}]|{r["status"]}|\n'
text+=f'\n主测{primary_n}请求、额外非预定重复{extra_n}请求，后者单独保留不替换主配对。每域2个ABBA区块/8独立进程/4配对；Student-t logratio df3，2负载×2指标Bonferroni，近似过程分布假设与小配对数限制明确。不是p>0.05作为等效证据。各请求/进程全部保留。\n\n'
text+='**旧E4机制**：FullV计划服务利用率约1.004–1.254，排队是总延迟放大的直接算术来源。HR服务36.05→45.03ms与queue4.22→295.15ms可复算总40.55→340.41ms。不能把25%服务差异解释为排队本身的原因，旧GPUclock/CPU抢占/worker状态missing；无法确认单一根因。新常驻worker/实际线程预热/固定affinity/低负载作为共同工程修复，未做单因素消融。新CPU/GPU telemetry只支持关联。\n\n'
text+='**资源与工程披露**：4个初始空闲检查未过的attempt在CUDA前终止；2个进程被ResourceGuard外来PID污染中止（foreign1506944、1527619，身份无法事后确认），失败日志与已有partial/telemetry保留，不参与有效主对照。恢复阶段只补未完成/污染预定槽位，不重测健康槽位追求通过。Finance恢复扫描误把仍在运行的槽位当缺失，导致一次额外有效测量，156条全部单独保留；主槽位始终采用原定第一有效attempt002，未根据结果选择。其后尝试取消额外进程的脚本因已有测量触发保护断言，没有发出信号、未终止任何进程。旧worker在异常前部分内存行未逐条写出，此缺失不能补造；修订worker在测量格之间保存partial并在finally保存已完成invoke。\n\n'
text+='**中/近饱和**：仅轻负载整体PASS的域可执行预定.65/.95参考服务率的4独立进程ABBA。'+str(len(stress))+'个格已完成；具体QPS/queue/service/吞吐见medium_near_load_diagnostics.csv。该负载诊断不替换轻负载主门槛，近饱和排队/失稳不能作普遍吞吐主张。\n\n'
text+='**决策**：仅'+','.join(passed)+'在2/5QPS同协议比较冻结路由；未过域的中/近负载和路由流式标NOT_RUN，原因是可靠性门槛未满足。没有将单个已通过指标或2QPS部分通过作为切换整体门槛的理由。S2继续独立的非流式常驻单请求真实成本对照，不能写成流式净收益。\n\n'
text+='**论文支撑**：Serving Reliability图/表、旧E4故障分解、完整负结果与修复披露。当前不支持跨四域Serving优越性、不支持证明唯一CPU/GPU根因。\n'
C.prose(C.R/'S1_decision.md',text);C.prose(C.R/'round_summary.md','# 结论—决策—阻塞\n\n'+text)
facts=f'新预登记轻负载A/A主测{primary_n}请求，8独立进程/域、4配对；PASS域：{passed}；其余INSUFFICIENT等效性，不进入策略流式。旧E4队列放大分解已复算；外来PID污染2次均中止并保留；Finance额外156条误重复透明保留、不替换主结果。'
C.finish('PASS_DIAGNOSIS_PARTIAL_AA_EQUIVALENCE',facts,'只在'+','.join(passed)+'已过2/5QPS门槛时按冻结协议测流式；其他域暂停Serving优越性主张。S2补充独立常驻完整路径成本，不依赖流式门槛。')

