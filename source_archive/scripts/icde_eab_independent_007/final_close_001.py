"""Immutable round closure; never changes old scientific products or protocols."""
import common as C
import json,csv,shutil,subprocess,datetime,os
from pathlib import Path
C.start('final_001')
stages={'T0':'T0','T1':'T1','T3_pilot':'T3/pilot_summary_001','T2':'T2/summary_001','T3_expansion':'T3/expansion_summary_001','T4':'T4/rebuild_004','T5':'T5/integration_002'}
expected={'T0':'f909a6cec47c208a61d080ca61f2a25a9cecd4218375e646efc4e9447700ba8b','T1':'d0b1ebde068ccc17c7f5b02d90139f3f90966fdd10096094585e48e887b673fe','T3_pilot':'64e23885a0ea0ad090ca7ccc2baad9d9227ea5390791c95555546d763808f164','T2':'642c50223f78a2bf59fc168e2ccbee2b4bd240a02245633d7db30ebc4037eeb8','T3_expansion':'e7961fc76408827cef4f39ff93761b0d617f9dcea4bfb7453b6a857532a73e3e','T4':'7c94414519686764bd752e0eb1114498e46c49e3a08e467d53ad6096934bf17a','T5':'a48f2ca6756e3bb7febfb2a39d897178bf6d20ba5650536028839947b1f3cb71'}
records=[]
for name,stage in stages.items():
    p=C.ART/stage/'manifest.json';assert C.sha(p)==expected[name],name
    m=C.load(p,'CLOSED_STAGE_MANIFEST');inventory=C.REPORT/stage/'hash_inventory.json';assert C.sha(inventory)==m['inventory_SHA256']
    inv=C.load(inventory,'CLOSED_STAGE_HASH_LEDGER')
    for path,h in inv['outputs'].items():assert C.sha(C.ROOT/path)==h,'CLOSED_OUTPUT_CHANGED '+path
    records.append({'stage':name,'path':str(p.relative_to(C.ROOT)),'manifest_SHA256':C.sha(p),'status':m['status'],'time_bj':m['time_bj'],'code_snapshot_commit':m['code_snapshot_commit']})
    print(json.dumps({'closed_stage_verified':name,'bound_files':len(inv['outputs']),'time_bj':C.stamp()}),flush=True)
proto=C.load(C.ART/'T1/biomedical_frozen_protocol.json','FROZEN_NEW_BENCHMARK_PROTOCOL')
assert C.sha(C.ART/'T1/biomedical_frozen_protocol.json')=='5f3db14fea91d12128d7507d07d7a3c9fe54a7c1dea3d3354207eea84357be3b'
for p,h in proto['code_SHA256'].items():assert C.sha(C.ROOT/p)==h,'FROZEN_CODE_CHANGED '+p
freeze=C.ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
check005=[]
for name,h in {'protocol.json':'ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41','asset_lock.json':'116b3a4a486df6049f05221f9cb4856ab5c808605bfa6659d4fca530a9848bbe','freeze_record.json':'f276af09e40cad07c6ad180fc3a8ee8250e6565518b1db865b0f81178a465004'}.items():
    actual=C.sha(freeze/name);assert actual==h;check005.append({'file':name,'sha256':actual,'status':'PASS_UNCHANGED_METADATA_ONLY'})
C.table(C.R/'frozen005_final_recheck.csv',check005)
C.table(C.REPORT/'run_timeline.csv',records);C.write(C.REPORT/'stage_manifest_index.json',records)
copies={
'T1/biomedical_provenance_audit.md':'biomedical_provenance_audit.md',
'T1/biomedical_frozen_protocol.yaml':'biomedical_frozen_protocol.yaml',
'T1/biomedical_query_family_map_002.csv':'biomedical_query_family_map.csv',
'T2/summary_001/biomedical_main_table.csv':'biomedical_main_table.csv',
'T2/summary_001/biomedical_per_query.csv':'biomedical_per_query.csv',
'T2/summary_001/biomedical_failure_cases.md':'biomedical_failure_cases.md',
'T2/summary_001/biomedical_setup_amortization.csv':'biomedical_setup_amortization.csv',
'T2/summary_001/gpu_cost_breakdown.csv':'biomedical_gpu_cost_breakdown.csv',
'T3/expansion_summary_001/cross_retriever_main_table.csv':'cross_retriever_main_table.csv',
'T3/expansion_summary_001/gpu_cost_breakdown.csv':'colpali_gpu_cost_breakdown.csv',
'T3/expansion_summary_001/setup_cost.csv':'colpali_setup_cost.csv',
'T3/expansion_summary_001/retriever_specific_failure_cases.md':'colpali_failure_cases.md',
'T4/rebuild_004/eab_main_results.csv':'eab_main_results.csv',
'T4/rebuild_004/eab_cross_domain_cross_retriever.csv':'eab_cross_domain_cross_retriever.csv',
'T4/rebuild_004/eab_oracle_gap_decomposition.csv':'eab_oracle_gap_decomposition.csv',
'T4/rebuild_004/eab_baseline_fidelity.md':'eab_baseline_fidelity.md',
'T4/rebuild_004/primary_reference_parity.csv':'primary_reference_parity.csv',
'T5/integration_002/artifact_readme.md':'artifact_readme.md',
'T5/integration_002/EAB_go_nogo.md':'decision_gate.md'}
for src,dst in copies.items():
    target=C.REPORT/dst;assert not target.exists();shutil.copyfile(C.REPORT/src,target)
shutil.copytree(C.REPORT/'T4/rebuild_004/figures',C.REPORT/'figures')
matrix=C.load(C.REPORT/'T4/rebuild_004/eab_claim_to_evidence_matrix.md','CANONICAL_CLAIM_MATRIX')
matrix+='\n\n## Final manuscript and artifact evidence\n\nPASS: native Paper R7 /11 rendered pages /13 input hashes verified;15 numeric CSV hashes replay identical. Bio tables are descriptive frozen-policy evaluation; ColPali rows are exposed Dev with own reference.\n\nINSUFFICIENT: strict Serving equivalence, public artifact accessibility, portable clean-checkout validation and final author/ORCID/AI disclosure. NOT_CERTIFIED: no new K1. No sealed data were used.\n\nReporting clarification: T4 stage facts say "32新ColPali行"; the actual table has16 new ColPali plus16 historical primary-model static rows,32 cross-model comparison rows. CSVs and all numbers are unchanged; do not count32 new ColPali observations.\n'
C.prose(C.REPORT/'claim_to_evidence_matrix.md',matrix);C.prose(C.REPORT/'eab_claim_to_evidence_matrix.md',matrix)
bio=C.load(C.REPORT/'biomedical_main_table.csv');by={x['method']:x for x in bio};cross=C.load(C.REPORT/'cross_retriever_main_table.csv');main=C.load(C.REPORT/'eab_main_results.csv');gaps=C.load(C.REPORT/'eab_oracle_gap_decomposition.csv')
assert len(bio)==9 and len(cross)==16 and len(main)==49 and len(gaps)==46
parity=C.load(C.REPORT/'primary_reference_parity.csv');assert len(parity)==656 and all(x['utility_risk_parity']=='True' for x in parity)
replay=C.load(C.ART/'T5/replay_003/replay_result.json');assert replay['status']=='PASS' and len(replay['checks'])==15
counts={}
counts['Bio_unprofiled_requests']=sum(len(C.load(C.REPORT/f'T2/process_{i}_001/requests.csv','NEW_ACTUAL_UNPROFILED_TRACE')) for i in range(3));assert counts['Bio_unprofiled_requests']==12960
counts['ColPali_expansion_unprofiled_requests']=0
canonical=[]
for domain in C.DOM:
    suffix='006' if domain=='Pharmaceuticals' else '004'
    for i in range(3):
        stage=f'T3/expansion_{domain}_p{i}_{suffix}';counts['ColPali_expansion_unprofiled_requests']+=len(C.load(C.REPORT/stage/'requests.csv','NEW_ACTUAL_DEV_TRACE'));canonical.append(stage)
assert counts['ColPali_expansion_unprofiled_requests']==5904
counts.update({'Bio_queries':160,'Bio_families':157,'Bio_source_groups':27,'ColPali_exposed_Dev_queries':164,'ColPali_pilot_queries_subset_of_Dev':24,'primary_parity_query_plan_pairs':656,'independent_query_count_increased_by_timing_repeats':0})
C.write(C.REPORT/'actual_execution_counts.json',counts)
# Retain and index failures. Never alter failed files or mask them with success summaries.
failures=[]
for p in sorted(C.ART.rglob('failure.json')):
    x=json.loads(p.read_text(encoding='utf8'));failures.append({'path':str(p.relative_to(C.ROOT)),'time_bj':x.get('time_bj'),'exception':x.get('exception'),'status':'FAIL_RETAINED'})
C.table(C.REPORT/'failure_inventory.csv',failures,fields=['path','time_bj','exception','status'])
engineering='''# Failure and engineering disclosures

All original failed/busy/aborted attempts and intermediate reports remain. The CSV inventories application exception records; pre-CUDA busy checks, own-job CPU abort receipts, syntax/transport errors and native Paper revisions are also retained at their original paths.

* ColPali001: missing BM25 CFG; aborted before a valid experiment.002: official-score mismatch. Diagnostic144 pairs found4 negative token maxima, max score error0.0327529907.004/HR005 fixes only true-padding masking/negative maxima; tolerance1e-3 unchanged. The correction preceded Biomedical outcomes;005 historical artifacts unchanged.
* T1 serial CPU audit was stopped by its owner and rerun with four CPU OCR workers; no other user's job was stopped. Review001 missed a Pharma alias;002 repairs the audit path. All62 image candidates visually reviewed; image-semantic encoder audit NOT_RUN. Empty OCR causes4720 text-screen collisions; raw candidates retained.
* Pharma ColPali expansion004 failed on role filter `dev`; the actual approved literal is `development`.006 adds exact45 Dev-ID assertion, reuses the valid unlabeled index, retains all failed attempts. It does not repartition or tune.
* Expansion summary006 used unsupported Python3.12 f-string syntax on server3.11.007 corrects quoting only. Raw GPU data unchanged.
* Initial T4 report and rebuild004 retained. Final root tables use rebuild004. Its stage-facts phrase32 new ColPali rows is clarified to16 new ColPali plus16 old primary rows. Actual CSV cells are unchanged. Root family map is canonical157 groups; older T1 map with160 provisional singleton groups is historical, not the frozen grouping.
* Primary scorer parity001 could not admit on busyGPU4;002 on available3/5 agrees in all656 utility/risk pairs. Not a latency equivalence result.
* Native PaperR5/R6/R7 builds retained. Visual QA found a duplicated method-tail in R6 from a bounded preimage read. R7 uses complete verified preimage and removes that duplicate only. Windows pdftotext default encoding failed a local QA read; explicit UTF-8 output succeeded. Final11 pages visually inspected.

None of these changes refits a policy, selects a benchmark threshold, opens sealed labels, replaces frozen005, or deletes unfavorable science. GPU dummy memory occupation was not used. ResourceGuard and three consecutive idle checks govern actual jobs; no other process was killed and shared .venv was unchanged.
'''
C.prose(C.REPORT/'failures_and_repairs.md',engineering)
reads=[]
for root in [C.ART,C.ROOT/'artifacts/icde_eab_independent_007_operations_001']:
    for p in root.rglob('*access*ledger*.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():reads.append({'log':str(p.relative_to(C.ROOT)),**json.loads(line)})
C.write(C.REPORT/'access_boundary_audit.json',{'time_bj':C.stamp(),'scope':'application guarded reads and explicit official-Biomedical ledgers; not a whole-host OS isolation proof','sealed_reads':{'Pharma_Cal81':0,'Pharma_Eval73':0,'historical_BL_raw':0,'old_test':0,'CS':0},'new_Biomedical_public_reads_not_project_sealed_test':len(reads),'ledgers':sorted({x['log'] for x in reads}),'frozen005_metadata_hash_recheck':'PASS','new_certification':'NOT_CERTIFIED'})
C.prose(C.REPORT/'round_summary.md','# icde_eab_independent_007｜结论—决策—阻塞\n\n实际封账时间：'+C.stamp()+'（北京时间）。\n\n**结论。** PASS：新Biomedical全部160英文/157查询组/27来源完成冻结策略评价；9方法12960次完整常驻路径计时，3独立进程×3重复。ColPali24问正确性pilot后完成四域164 Dev、4静态计划5904次计时。均不是新增正式认证。\n\n|Biomedical|nDCG@10|正遗憾r（来源95%区间）|实际E2E ms|\n|---|---:|---|---:|\n'+'\n'.join('|'+name+'|'+f'{float(by[k]["nDCG10"]):.4f}'+'|'+f'{float(by[k]["mean_r"]):.4f} [{float(by[k]["mean_r_source95_low"]):.4f},{float(by[k]["mean_r_source95_high"]):.4f}]'+'|'+f'{float(by[k]["E2E_mean_ms"]):.2f}'+'|' for k,name in [('Full-Visual','FullV'),('BM25-R50','R50'),('BM25-R100','R100'),('Q_regret','Q'),('QP_regret','QP')])+'\n\n**决策。** EAB分析方向条件继续；学习路由收益未证明。Bio同风险静态/随机包络比较Q−19.63pp、QP−14.05pp（区间均低于0）；post-hoc QP有质量信号但实际成本更高。保留所有域、负结果、失败记录；不增大模型，不调Bio阈值。\n\n**阻塞。** 预训练重叠UNKNOWN；新K1 NOT_CERTIFIED；严格Serving不足。公共artifact、可移植clean-checkout、作者/ORCID/AI披露仍pending，不能标投稿就绪。Cal/Eval、旧B-L/test/CS读取0，005不变。\n\n**证据。** 本目录三张主表/三组图、claim矩阵、失败清单；原生PaperR7已编译、11页逐页检查，15数值CSV重放hash一致。总manifest/hash_inventory见同目录；T0–T5逐阶段已写Orange06 R61–R67（另含pilot/扩展记录）。完整成本分项、CPU/GPU及区间见CSV，不把活动GPU秒写成业务净收益。\n')
tree='''# 中文文件树与规范入口

项目根：/path/to/project/

reports/icde_eab_independent_007/
  round_summary.md                         一页结论—决策—阻塞
  decision_gate.md                         EAB条件继续，投稿尚未就绪
  eab_main_results.csv                     49行主表，逐行证据与成本等级
  eab_cross_domain_cross_retriever.csv      32行；16新ColPali+16旧主模型
  eab_oracle_gap_decomposition.csv          46行离线上界；非部署
  eab_baseline_fidelity.md                  adapted与官方方法的差异
  biomedical_main_table.csv                160问/9方法完整真实路径
  biomedical_per_query.csv                 逐查询效用/风险/成本
  biomedical_query_family_map.csv          规范冻结157组（源T1/..._002.csv）
  cross_retriever_main_table.csv            四域ColPali16静态行及区间
  primary_reference_parity.csv              656组合历史效用/风险一致
  figures/                                 三组PDF/PNG，可脚本重绘
  claim_to_evidence_matrix.md               主张—表图—证据等级
  failures_and_repairs.md                   修复与未完成项披露
  failure_inventory.csv                     保留失败索引
  artifact_readme.md                        环境/获取/模型索引/重放/许可证
  stage_manifest_index.json                 每阶段manifest/commit/hash
  hash_inventory.json, manifest.json         本轮总封账（不自包含循环hash）
  T0/                                      历史资产/边界冻结
  T1/                                      来源审计/协议，保留早期候选组
  T2/process_*_001/, summary_001/             原始Bio三进程与统计汇总
  T3/pilot_summary_001/                      24问先行正确性pilot
  T3/expansion_summary_001/                  四域164问扩展汇总
  T3/expansion_*_004/, *_006/                 原始执行，Pharma规范006
  T4/rebuild_004/                            规范主表/图，初稿T4保留
  T4_preparation/                            post-hoc探针/错误案例诊断
  T5/integration_002/                        原生Paper R7/PDF/工件说明
  T5/replay_003/                             实际CPU重放在artifacts同路径
  final_001/                                末次005核验与封账辅助

artifacts/icde_eab_independent_007/          原始guards/profiles/输入hash/manifest
  code_*/source.bundle                     各阶段私有Git代码快照
  code_final_001/source.bundle              最终源码快照，按approved_entrypoints执行
  T5/integration_002/paper_source_R7.zip     LaTeX权威源码工件
  T5/replay_003/replay_result.json           15数值表逐字节一致
scripts/icde_eab_independent_007/            版本化执行/统计/绘图脚本
cache/indexes/icde_eab_independent_007*/     新Bio与ColPali页向量（无封存标签）
artifacts/icde_eab_independent_007_operations_001/  已结束资源监测日志
artifacts/icde_eab_independent_007_logs_001/        原始执行与失败日志

旧run、冻结005、所有失败与中止产物均保留。该私有工件尚未外部发布。
'''
C.prose(C.REPORT/'中文文件树.md',tree)
C.write(C.R/'resource_final_snapshot.json',{'time_bj':C.stamp(),'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'compute_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),'disk_free_bytes':shutil.disk_usage(C.ROOT).free,'shared_venv_modified':False})
code=C.snapshot('final_001')
print(json.dumps({'final_hashing_started':C.stamp(),'code_commit':code['code_snapshot_commit']}),flush=True)
files={}
folders=[C.ART,C.REPORT,C.ROOT/'artifacts/icde_eab_independent_007_operations_001',C.ROOT/'artifacts/icde_eab_independent_007_logs_001',C.ROOT/'artifacts/icde_eab_independent_007_paper_transfer_001']
for folder in folders:
    if folder.exists():
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink() and p not in [C.ART/'manifest.json',C.ART/'hash_inventory.json',C.REPORT/'manifest.json',C.REPORT/'hash_inventory.json']:
                files[str(p.relative_to(C.ROOT))]=C.sha(p)
inventory={'run_id':C.RUN,'time_bj':C.stamp(),'files':files,'code':code['source_file_SHA256'],'external_assets':'T0/integrity_checks.csv and T0/model_hash_supplement_001 plus T5/integration_002/new_index_asset_ledger.csv; complete consumed index member hashes are in immutable index manifests','self_hash_policy':'Root hash_inventory and root manifest excluded to avoid cycles; stage files are included. Orange receipts/current status remain outside closed roots.'}
C.write(C.ART/'hash_inventory.json',inventory);C.write(C.REPORT/'hash_inventory.json',inventory)
manifest={'run_id':C.RUN,'status':'COMPLETED_ACTUAL_EXPERIMENTS_WITH_EXPLICIT_LIMITATIONS','time_bj':C.stamp(),'seed':C.SEED,'decision':'EAB_ANALYSIS_CONDITIONAL_GO_NOT_SUBMISSION_READY; LEARNED_ROUTER_ADVANTAGE_NOT_ESTABLISHED','stages':records,'actual_execution_counts':counts,'new_certification':'NOT_CERTIFIED','sealed_label_reads':{'Pharma_Cal81':0,'Pharma_Eval73':0,'historical_BL_raw':0,'old_test':0,'CS':0},'frozen005_unchanged':True,'frozen_biomedical_protocol_SHA256':C.sha(C.ART/'T1/biomedical_frozen_protocol.json'),'code_snapshot_commit':code['code_snapshot_commit'],'base_project_commit':code['base_project_commit'],'code_bundle_SHA256':code['bundle_SHA256'],'inventory_SHA256':C.sha(C.ART/'hash_inventory.json'),'bound_files':len(files),'paper_revision':7,'paper_build_id':'73e3058c-6456-4a8b-bc58-ff6000d6f836','paper_PDF_SHA256':'ccc4e905223449ca480d0a21e38e946338b1a2fe15c199badfa0440159b2b424','CPU_replay_tables_identical':15,'public_artifact':'NOT_PUBLISHED; CLEAN_CHECKOUT_PORTABILITY_NOT_RUN','strict_Serving':'INSUFFICIENT_PRIOR_AA_GATE; NO_NEW_SUPERIORITY_CLAIM','pretraining_independence':'UNKNOWN; UNUSED_IN_MDROUTER_SELECTION_QUALIFIED','reports':str(C.REPORT)}
C.write(C.ART/'manifest.json',manifest);C.write(C.REPORT/'manifest.json',manifest)
print(json.dumps({**manifest,'manifest_SHA256':C.sha(C.ART/'manifest.json')},ensure_ascii=False),flush=True)
