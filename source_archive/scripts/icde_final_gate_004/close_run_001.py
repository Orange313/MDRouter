import common as C
import json,pathlib,os,subprocess,hashlib,shutil
__import__('sys').addaudithook(C.protected);checks=[];stages={};known=[]
permitted='artifacts/icde_final_gate_004/S0/asset_binding_001/input_reads.jsonl'
for s in ['S0','S1','S2','S3','S4']:
 m=C.load(C.ART/s/'manifest.json');iv=C.load(C.REPORT/s/'hash_inventory.json');assert C.sha(C.REPORT/s/'hash_inventory.json')==m['inventory_SHA256'];mismatches=[]
 for p,h in iv['outputs'].items():
  actual=C.sha(C.ROOT/p)
  if actual!=h:
   assert p==permitted,('Unexpected scientific output changed',p)
   raw=(C.ROOT/p).read_bytes();chunks=raw.splitlines(keepends=True);prefix=b'';match=None
   for j,ch in enumerate(chunks,1):
    prefix+=ch
    if hashlib.sha256(prefix).hexdigest()==h:match={'old_prefix_lines':j,'old_prefix_bytes':len(prefix)};break
   assert match is not None,'Expected append-only metadata audit-log change, not a rewrite'
   known.append({'path':p,'old_SHA256':h,'final_SHA256':actual,**match,'final_lines':len(chunks),'reason':'binding ran in parallel underS0; source read-audit appended index_manifest metadata after mainS0 inventory, no scientific data/code/model changed','status':'APPEND_ONLY_AUDIT_LOG_SEALED'})
   mismatches.append(p)
 # Source scripts are immutable by version; additions are allowed, no edits to bound versions.
 code_record=C.load(C.ART/('code_'+s)/'code_commit_record.json')
 for p,h in code_record['source_file_SHA256'].items():assert C.sha(C.ROOT/p)==h,('Bound source version changed',s,p)
 stages[s]={'status':m['status'],'manifest_SHA256':C.sha(C.ART/s/'manifest.json'),'time_bj':m['time_bj'],'code_snapshot_commit':m['code_snapshot_commit']};checks.append({'stage':s,'output_checks':len(iv['outputs']),'unexpected_scientific_mismatches':0,'known_append_only_metadata_log':mismatches,'status':'PASS_SCIENTIFIC_INTEGRITY'})
seal=C.ART/'S0_binding_closure_002';seal.mkdir(exist_ok=False);C.write(seal/'append_only_audit_log_seal.json',known);binding=C.load(C.ART/'S0/asset_binding_001/manifest.json');assert C.sha(C.REPORT/'S0/asset_binding_001/asset_verification.csv')==binding['verification_csv_SHA256'];C.write(seal/'manifest.json',{'time_bj':C.stamp(),'status':'PASS','known_async_log_append_only':known,'binding_assets':binding['asset_count'],'binding_manifest_SHA256':C.sha(C.ART/'S0/asset_binding_001/manifest.json'),'scientific_output_mismatches':0})
for n,h in C.EXPECTED.items():assert C.sha(C.FROZEN/n)==h
assert C.sha(C.PREV/'manifest.json')=='0fdb139301921255cd82214fb4bb59f36d63ad8fbd8068a46d352b28b201eb73'
C.table(C.REPORT/'integrity_verification.csv',checks)
actual=C.load(C.REPORT/'S2/resident_analysis_002/static_vs_router_matched_risk.csv');gate=C.load(C.ART/'S1/AA_gate.json');streams=C.load(C.REPORT/'S2/streaming_status_all_domains.csv');repro=C.load(C.REPORT/'reproduction_verification_001/manifest.json');facts=[]
for d in C.DOM:
 r=next(x for x in actual if x['domain']==d and x['method']=='QP_regret');facts.append(f'|{d}|{float(r["mean_r"]):.5f}|{float(r["matched_risk_increment_pp"]):.2f}|[{float(r["paired95_increment_low_pp"]):.2f},{float(r["paired95_increment_high_pp"]):.2f}]|{r["resident_AA_status"]}|')
text='# icde_final_gate_004｜结论—决策—阻塞\n\n**结论**：已实际完成S0–S4及必要失败分支。新方法收益INSUFFICIENT；Regular NO_GO；选择EAB分析主线，但当前尚未投稿就绪。\n\n'
text+='- PASS：输入/历史封账/005/模型和索引核验；四域164Dev、8方法实际完整路径11,808次计时、1,312GPU profiles。\n- PASS/INSUFFICIENT分开：新轻负载A/A主测5,248请求、8独立进程/域；HR/Pharma通过，Finance/Industrial未建立±10%等效。额外156条误重复保留，不替换主对照；两通过域中/近饱和16格均保留。\n- INSUFFICIENT：后续流式被外来GPU PID中止，HR仅1完整进程(1,280请求)、Pharma0；不可宣称稳定Serving/Goodput收益。\n- PASS：168组既有训练预测重现、12训练子集嵌套；不支持稳定增样改善，因果INCONCLUSIVE。全7κ/3seed保留。\n- NOT_CERTIFIED / NOT_RUN：新K1、独立Eval、完整业务净收益；PharmaCal/Eval及旧B-L/test/CS读取0。\n\n'
text+='|域|QP mean r|相对同风险静态随机包络完整路径增量pp|条件paired95%|常驻A/A|\n|---|---:|---:|---|---|\n'+'\n'.join(facts)+'\n\n所有点为Dev条件探索。仅Finance新常驻A/A通过，其他域的成本优越性未建立；即使Finance也没有可靠路由增量。静态随机LP是理想期望包络，实际冻结随机路由含执行决策成本；流式latency非线性，不把插值当实际混合Serving结果。\n\n'
text+='**决策**：不扩大LoRA/CMRP/RMix/PPI；论文围绕强静态计划、信息/风险/成本前沿与测量边界整理。Eval73只生成未冻结草案，未来必须单独授权，Cal及旧测试不随之解封。正式风险合同保持ε=.05/δ=.01。\n\n'
text+='**阻塞**：跨进程/硬件噪声、外来短进程污染、独立抽样与功效不足、公开artifact未发布。所有原始请求/trace/失败保留。新风险保证与全业务净收益不成立。\n\n'
text+=f'**证据**：S0–S4/、claim_to_evidence_matrix.md、integrity_verification.csv、REPRODUCE.md、中文文件树.md。CPU复算{repro["tables"]}张核心CSV SHA逐字节一致；完整全硬件复现NOT_RUN。总manifest与hash_inventory.json单独提供，避免自引用。\n'
C.prose(C.REPORT/'round_summary.md',text)
tree='# 中文文件树（本轮实际目录）\n\n```text\n/path/to/project/\n├── scripts/icde_final_gate_004/             # 版本化脚本、失败版本和复算入口\n├── reports/icde_final_gate_004/\n│   ├── authority/                         # Orange06 R44、10/12/13快照\n│   ├── S0/                                # 输入冻结、5829项模型/索引资产复核\n│   ├── S1/                                # 旧E4故障分解、新A/A、负载诊断\n│   ├── S2/\n│   │   ├── resident_analysis_001/          # 聚合轴错误；失败保留\n│   │   ├── resident_analysis_002/          # 实际完整路径主表/前沿/Oracle/False-Safe\n│   │   └── *_stream_*.csv                  # 有效完整进程、污染局部、NOT_RUN\n│   ├── S3/                                # 训练组成、风险校准、全部梯度/来源敏感性\n│   ├── S4/                                # 论文表图、证据矩阵、Eval草案、投稿判断\n│   ├── figures_v2_001/                    # 仅修正A/A图颜色图例，数值未变\n│   ├── reproduction_verification_001/     # 23张CSV复算hash一致\n│   ├── round_summary.md / REPRODUCE.md\n│   └── manifest.json / hash_inventory.json\n└── artifacts/icde_final_gate_004/\n    ├── S0…S4/                             # 模型/协议、逐请求、GPU traces、guard、manifest\n    ├── S0_binding_closure_002/             # 并行只追加audit日志最终seal\n    ├── reproduce_*_001/                   # CPU独立复算结果与manifest\n    ├── code_S0…code_S4/                   # 分阶段隔离Git快照/bundle\n    ├── logs/                              # 所有中止、失败、恢复日志\n    └── code_final_001/                     # 最终代码快照\n```\n\nCPU/GPU原始trace较大留服务器；本地报告包另有下载校验回执。最终Orange回执在独立companion目录，避免封账hash循环。\n'
C.prose(C.REPORT/'中文文件树.md',tree);C.prose(C.REPORT/'REPRODUCE.md',C.load(C.REPORT/'S4/reproduction_materials_checklist.md'))
C.prose(C.REPORT/'archive_disclosures.md','# 归档与失败披露\n\nS0模型/索引binding与主hash扫描并行，导致其input_reads.jsonl在主manifest后追加一条无标注索引元信息读取；已证明旧SHA对应最终日志完整前缀，最终seal独立绑定新bytes。没有科学数据/模型/代码变更。旧manifest保留，所有最终产物由本轮总inventory绑定。\n\n旧S1图自动颜色循环与QPS图例不一致，数值未变；figures_v2_001修正样式，旧图保留。resident_analyze_001维度轴错误在生成表前失败，002修复并验证形状，23表独立CPU复算一致。Finance额外误启动有效重复全部保留，主分析只使用第一有效预定槽位；保护断言阻止对已测进程发送取消信号，实际未终止任何他人/自己作业。资源中止按whole-process失效，不挑之前有利的格。\n')
code=C.snapshot('final_001');C.write(C.REPORT/'final_code_commit_record.json',code)
files={str(p.relative_to(C.ROOT)):C.sha(p) for folder in [C.ART,C.REPORT,C.ROOT/'scripts'/C.RUN] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p not in [C.ART/'manifest.json',C.ART/'hash_inventory.json',C.REPORT/'manifest.json',C.REPORT/'hash_inventory.json']}
iv={'time_bj':C.stamp(),'files':files,'note':'scientific tree closed; final Orange/status/download receipts are outside closedtree to avoid circular SHA'};C.write(C.ART/'hash_inventory.json',iv);C.write(C.REPORT/'hash_inventory.json',iv)
m={'run_id':C.RUN,'status':'COMPLETED_WITH_LIMITATIONS','time_bj':C.stamp(),'stages':stages,'inventory_SHA256':C.sha(C.ART/'hash_inventory.json'),'code_snapshot_commit':code['code_snapshot_commit'],'code_bundle_SHA256':code['bundle_SHA256'],'base_git_commit':code['shared_project_HEAD_unchanged'],'predecessor_manifest_SHA256':'0fdb139301921255cd82214fb4bb59f36d63ad8fbd8068a46d352b28b201eb73','frozen005_protocol_SHA256':C.EXPECTED['protocol.json'],'S1_AA_PASS_domains':[d for d,v in gate['domains'].items() if v['status']=='PASS'],'actual_resident_requests':11808,'GPU_profiles':1312,'streaming_status':streams,'CPU_reproduction_tables_identical':repro['tables'],'new_formal_certification':'NOT_CERTIFIED','independent_Eval':'NOT_RUN_SEALED','sealed_Cal_Eval_oldBL_test_CS_reads':0,'old_protocol_modified':False,'decision':'REGULAR_NO_GO_EAB_ANALYSIS_CONDITIONAL_NOT_SUBMISSION_READY','unexpected_scientific_hash_mismatches':0,'known_append_only_audit_log_seal_SHA256':C.sha(seal/'manifest.json')};C.write(C.ART/'manifest.json',m);C.write(C.REPORT/'manifest.json',m);print(json.dumps({'status':m['status'],'manifest_SHA256':C.sha(C.ART/'manifest.json'),'code_snapshot_commit':code['code_snapshot_commit'],'code_bundle_SHA256':code['bundle_SHA256'],'time_bj':m['time_bj'],'files':len(files)}),flush=True)
