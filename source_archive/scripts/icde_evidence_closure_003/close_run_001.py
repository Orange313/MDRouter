import common as C
import json,pathlib,collections,os,subprocess
R=C.REPORT;A=C.ART;checks=[];stages={}
for s in ['E0','E1','E2','E3','E4','E5']:
 m=json.loads((A/s/'manifest.json').read_text());iv=json.loads((R/s/'hash_inventory.json').read_text());assert C.sha(R/s/'hash_inventory.json')==m['inventory_SHA256'];bad=[]
 for rel,h in iv['outputs'].items():
  actual=C.sha(C.ROOT/rel)
  if actual!=h:bad.append(rel)
 permitted=[str((A/'E2'/f).relative_to(C.ROOT)) for f in ['learning_driver_001.log','pipeline_driver_001.log']] if s=='E2' else []
 assert set(bad)<=set(permitted),(s,bad)
 for rel,h in m['code_SHA256'].items():assert C.sha(C.ROOT/rel)==h,('Code changed',s,rel)
 checks.append({'stage':s,'declared_output_n':len(iv['outputs']),'unexpected_hash_mismatch_n':0,'known_live_stdout_log_changes':bad,'closure':'E2/closure_seal_001 verifies final logs; scientific outputs unchanged' if bad else 'PASS'});stages[s]={'status':m['status'],'manifest_SHA256':C.sha(A/s/'manifest.json'),'time_bj':m['time_bj']}
seal=json.loads((A/'E2/closure_seal_001/closed_stage_output_inventory.json').read_text())
for rel,h in seal.items():assert C.sha(C.ROOT/rel)==h,('Closed E2 output changed',rel)
for name,h in C.EXPECTED.items():assert C.sha(C.FROZEN/name)==h
C.table(R/'integrity_verification.csv',checks)
summary=json.loads((A/'E5/manifest.json').read_text());code=json.loads((R/'E5/code_commit_record.json').read_text());static=list(__import__('csv').DictReader((R/'E1/static_quality_e2e.csv').open()));stream=list(__import__('csv').DictReader((R/'E4/streaming_window_sweep.csv').open()));matches=list(__import__('csv').DictReader((R/'E4/matched_risk_e2e.csv').open()));matched=[x for x in matches if x['status']=='EXPLORATORY_MATCHED_POINT']
text='# icde_evidence_closure_003｜结论—决策—阻塞\n\n'
text+='**本轮结论**：PASS 完成E0–E5：旧三域119 Dev七计划真实成本；Pharma Fit81/Dev45官方636条qrels、504条逐计划结果及756组训练/迁移对照。R100相对FullV常驻E2E节省HR9.08%[8.12,9.97]、Finance20.63%[19.82,21.48]、Industrial34.22%[33.07,35.56]；但原R50伤害修复仅1/6、2/5、0/4，主要伤害修复门槛FAIL。增加训练样本没有稳定改善证据，Probe增量INSUFFICIENT。\n\n'
text+=f'**决策**：E4完成{len(stream)}个工作点、{sum(int(x["request_n"]) for x in stream)}条真实流式主测量请求；{len(matched)}/{len(matches)}组与随机基线在点估计上满足风险差<=.005，其余保留INSUFFICIENT。全部逐点描述区间和A/A均在E4，不将多配置开发结果写成独立显著性。当前优先EAB分析路线，Regular方法增量不足；不扩大模型。保留FullV/BM100/BM25最小候选，BM50、R200、Dense、Hybrid的静态负结果/前沿均保留。\n\n'
text+='**阻塞**：新认证NOT_CERTIFIED。Cal81的理想单策略全零UCB=.05527>.05，且IID尚未证明；Cal/Eval继续封存。新的独立查询抽样/标注预算和未来揭封需另行决定。Serving为包含窗口/队列/特征/Probe的进程内真实执行，未计网络与完整业务能耗；不能将GPU节省写成完整净收益。\n\n'
text+=f'**证据**：各阶段目录E0–E5；核心报告E5/main_experiment_summary.md，中文文件树与integrity_verification.csv，所有输入/代码/模型/索引/输出见hash_inventory.json。代码快照commit `{code["code_snapshot_commit"]}`；共享项目base commit `{code["shared_project_HEAD_unchanged"]}`未改变。全部历史与失败日志保留。\n'
C.prose(R/'round_summary.md',text)
tree='# 中文文件树（实际生成）\n\n```text\n/path/to/project/\n├── scripts/icde_evidence_closure_003/  # 本轮版本化脚本\n├── reports/icde_evidence_closure_003/\n│   ├── authority/  # 开始时Orange06/10/11/12快照\n│   ├── E0/  # 冻结输入、分区与指标口径\n│   ├── E1/  # 119Dev七计划实测与候选遗漏\n│   ├── E2/  # Pharma合法数据、学习/迁移、配对区间及封账\n│   ├── E3/  # 独立性/功效审计；NOT_CERTIFIED\n│   ├── E4/  # 真实流式、风险匹配、A/A、GPU独立重放\n│   ├── E5/  # 核心表图、论文判断、复现与工程披露\n│   ├── round_summary.md\n│   ├── claim_to_evidence_matrix.md\n│   ├── manifest.json / hash_inventory.json\n│   └── integrity_verification.csv\n├── artifacts/icde_evidence_closure_003/E0…E5/  # 原始请求/trace、模型、manifest、Git bundle\n├── cache/indexes/pharma_fixedcorpus_index_001/  # 新2313页固定语料索引\n└── data/primary/vidore_v3_pharmaceuticals/262e203…/corpus/  # 两个官方无标注分片\n```\n\n'
for s in ['E0','E1','E2','E3','E4','E5']:
 n=sum(p.is_file() for p in (A/s).rglob('*'));nr=sum(p.is_file() for p in (R/s).rglob('*'));tree+=f'- {s}：artifacts实际{n}个文件，reports实际{nr}个文件。\n'
C.prose(R/'中文文件树.md',tree)
statuspath=C.ROOT/'manifests/current_experiment_status.json';oldbytes=statuspath.read_bytes();old=json.loads(oldbytes)
with (A/'status_prior_close_raw_001.json').open('xb') as f:f.write(oldbytes)
updated=dict(old);updated['icde_evidence_closure_003']={'status':'COMPLETED_WITH_LIMITATIONS_NOT_CERTIFIED','completed_time_bj':C.stamp(),'stage_status':stages,'new_certification':False,'Cal_Eval_oldBL_test_CS_label_reads_this_run':0,'decision':'EAB_PRIORITY_REGULAR_INSUFFICIENT','report_root':str(R),'code_snapshot_commit':code['code_snapshot_commit']};updated['latest_run_id']=C.RUN
C.write(C.ROOT/'manifests/icde_evidence_closure_003_status_001.json',updated);tmp=statuspath.with_name('current_experiment_status.icde_evidence_closure_003.tmp.json');C.write(tmp,updated);assert statuspath.read_bytes()==oldbytes,'Concurrent status changed; preserve temp, do not overwrite';os.replace(tmp,statuspath);C.write(A/'final_status_snapshot_001.json',updated)
files={str(p.relative_to(C.ROOT)):C.sha(p) for folder in [A,R,C.ROOT/'scripts'/C.RUN] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p not in [A/'manifest.json',A/'hash_inventory.json',R/'manifest.json',R/'hash_inventory.json']};inventory={'time_bj':C.stamp(),'files':files,'closure_note':'E5 Orange receipt is archived outside this closed tree to avoid circular self-hashing; old E2 primary live-log hashes superseded only for logs by append-only seal'};C.write(A/'hash_inventory.json',inventory);C.write(R/'hash_inventory.json',inventory)
manifest={'run_id':C.RUN,'status':'COMPLETED_WITH_LIMITATIONS_NOT_CERTIFIED','time_bj':C.stamp(),'stages':stages,'inventory_SHA256':C.sha(A/'hash_inventory.json'),'code_snapshot_commit':code['code_snapshot_commit'],'base_project_commit':code['shared_project_HEAD_unchanged'],'frozen_005_protocol_SHA256':C.EXPECTED['protocol.json'],'Pharma_split_SHA256':'a70d0eec64f9611b0371a0f74fcba389f194798fd7b27d5e8e141b71f495440e','Pharma_index_SHA256':C.sha(C.ROOT/'cache/indexes/pharma_fixedcorpus_index_001/index_manifest.json'),'new_Cal_Eval_oldBL_test_CS_label_reads':0,'formal_new_certification':'NOT_CERTIFIED','scientific_hypotheses_changed_to_pass':False,'failures_and_old_products_preserved':True,'cost_scope':'resident actual retrieval/harness E2E; no full business net claim','E2_live_log_final_seal_manifest_SHA256':C.sha(A/'E2/closure_seal_001/manifest.json'),'integrity_unexpected_mismatches':0,'paper_recommendation':'EAB_PRIORITY_REGULAR_INSUFFICIENT'};C.write(A/'manifest.json',manifest);C.write(R/'manifest.json',manifest);print(json.dumps({'status':manifest['status'],'manifest_SHA256':C.sha(A/'manifest.json'),'code_snapshot_commit':code['code_snapshot_commit'],'time_bj':C.stamp()},ensure_ascii=False),flush=True)
