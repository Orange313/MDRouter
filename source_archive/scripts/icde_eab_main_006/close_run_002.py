import common as C
import json,sys,subprocess,importlib.metadata as im
from pathlib import Path
stages={'X0':'X0','X1':'X1','X2':'X2/rebuild_002','X3':'X3','X4':'X4'}
records={};verified=[]
for name,path in stages.items():
    a=C.ART/path;r=C.REPORT/path;m=json.loads((a/'manifest.json').read_text());iv=json.loads((r/'hash_inventory.json').read_text());assert C.sha(r/'hash_inventory.json')==m['inventory_SHA256']
    for p,h in iv['outputs'].items():
        actual=C.sha(C.ROOT/p);verified.append({'stage':name,'file':p,'expected':h,'actual':actual,'status':'PASS' if h==actual else 'FAIL'});assert h==actual
    records[name]={'status':m['status'],'time_bj':m['time_bj'],'manifest':str((a/'manifest.json').relative_to(C.ROOT)),'manifest_SHA256':C.sha(a/'manifest.json'),'code_snapshot_commit':m['code_snapshot_commit']}
assert not [p for p in (C.ART/'X1').glob('pilot_*') if p.is_dir()],'GPU pilot outputs exist; dedicated outcome review required'
resource=json.loads((C.ART/'X1/resource_outcome.json').read_text());assert resource['polls']==134 and resource['idle_observations']==0
supp=json.loads((C.ART/'X2/cost_asset_supplement_003/result.json').read_text());assert supp['verified_assets']==5834
f=C.ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
assert C.sha(f/'protocol.json')=='ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41'
C.table(C.REPORT/'stage_integrity_recheck.csv',verified)
versions={x:im.version(x) for x in ['torch','transformers','colpali-engine','numpy','scipy','pandas','pyarrow']};versions['python']=sys.version
site=C.ROOT/'.venv/lib/python3.11/site-packages'
versions['critical_runtime_module_SHA256']={str(p.relative_to(site)):C.sha(p) for p in [site/'colpali_engine/models/paligemma/colpali/modeling_colpali.py',site/'colpali_engine/models/paligemma/colpali/processing_colpali.py',site/'transformers/models/paligemma/modeling_paligemma.py']}
C.write(C.ART/'environment_versions.json',versions)
summary='''# icde_eab_main_006｜结论—决策—阻塞

本轮执行的是文档16 R2；06 R54、14 R2、15实际R2为输入。所有时间为实际北京时间。

## 结论

|阶段|实际结果|状态|
|---|---|---|
|X0|2031历史产物、冻结005/split核验；5项封存拒绝测试；164 Dev/122候选族、Fit81；新矩阵80行仅计划|PASS|
|X1|不同家族ColPali训练模型/10资产核验；隔离兼容修复后254张量精确匹配、CPU合成编码；134轮/45分钟6卡全忙，GPU质量查询0|CPU审计PASS；GPU BLOCKED|
|X2|40行四域主表：32行历史11808完整路径计时+1312 profiles重建、8行RR-style适配组合成本；328项CPU推理x3重复；5834模型/索引资产一致；setup/Q曲线补齐|PASS CPU；完整RR路径INSUFFICIENT|
|X3|73 synthetic×10空跑及默认拒绝；160英语/1016页/27文档/515全score2 qrels官方新域审核，95多相关；与245合法FitDev/11609旧页未见本次exact/Jaccard覆盖的重复|审计PASS；独立性/真实接口INSUFFICIENT；新AA NOT_RUN|
|X4|既有Paper R3→R4，八节10页英文稿，32+8表、2图、权威编译与全部页面核验；7表CPU重放SHA一致|PASS工作稿；投稿INSUFFICIENT|

正式主表采用X2/rebuild_002/；X2根目录001是保留的工程中间版（错误成本配置键导致兼容性误判），不能作为最终结果。含历史共享setup的完整版本为X2/cost_asset_supplement_003/eab_main_table_dev_with_setup.csv；共享初始构建不等于BM25单用的增量索引成本。

RR-style hard/soft同风险增量(pp)：HR +6.35/+3.40、Finance -5.05/-7.60、Industrial -8.46/-14.27、Pharma -12.26/-19.95。配对区间见CSV；工作点平均r均>.05，不满足原预算。它们是固定Fit81同容量Ridge对照而非原版Qwen3/LoRA/KL复现；cost为组合估计，完整路径P95/CPU留missing。旧Q/QP动态方法收益仍未被证明。

## 决策

EAB继续有条件推进，当前NOT_READY；不扩大复杂模型。跨视觉检索器结果必须待锁定24-query GPU pilot；独立新域评价待训练/语义重叠审计与协议就绪。Pharma Eval真实接口尚不完整，不请求立即揭封。新K1 NOT_CERTIFIED，历史005不迁移。

## 阻塞与负结果

- 资源：45分钟134轮无空GPU，第二索引、第二模型GPU质量/成本、新AA与跨策略Serving均NOT_RUN；不得填0为成本或使用旧PASS替代。
- CPU模型加载发现旧adapter252个LoRA未正确映射；失败、部分JSON和dtype摘要失败均保留。新overlay仅改键，254张量数值不变并核对，原模型缓存和共享.venv不改。
- 新benchmark训练集重叠和语义图像近重复UNKNOWN；27来源组件不是IID n。真实Eval/Cal及旧B-L/test/CS读取0。
- 署名：旧legacy Orange连接绑定ChatGPT，本轮R55/R56和Paper R4由该连接署名。发现后停用，后续resource连接核验为CodeX；历史不改写，不自称真人批准。
- 复现：7表数值重放通过；原版RR、完整新GPU clean-checkout与匿名公开URL未完成。10页稿不是投稿ready。

## 证据

阶段manifest、raw审计、模型/索引hash、负结果和代码快照见artifacts/icde_eab_main_006/及本目录。Paper PDF R4 SHA256: 5ed1d6ce18196c1312e5a3d8a87ff619e6e1cf524477e43e4a2d0b7ee6b4be15；build 7fddf264-69bf-4bab-8efc-d3f414a5ea4a。全部阶段已分别写入Orange06；最终总manifest与回执由completion独立保存，避免循环hash。
'''
C.prose(C.REPORT/'round_summary.md',summary)
matrix=(C.REPORT/'X4/claim_to_evidence_matrix.md').read_text();matrix+='\n最终更新：X1 GPU BLOCKED，不能支持跨检索器发现；X3新域160英语只做官方输入/重叠审计，检索评价0且训练重叠未知；Pharma Eval73/Cal81封存，真实获取/硬件接口不完整。X2共享setup及5834资产重核见cost_asset_supplement_003。\n';C.prose(C.REPORT/'claim_to_evidence_matrix.md',matrix)
tree='''# 中文文件树（实际目录摘要）

```text
reports/icde_eab_main_006/
├── round_summary.md                  # 一页结论—决策—阻塞
├── claim_to_evidence_matrix.md        # 论文主张与证据等级
├── stage_integrity_recheck.csv        # 所有阶段输出重新hash
├── X0/                               # authority/执行协议/访问边界
├── X1/                               # 模型来源、24查询锁定、未执行GPU表与失败原因
├── X2/
│   ├── rebuild_002/                   # 正式40行Dev主表、配对区间、逐查询
│   └── cost_asset_supplement_003/     # 共享setup、Q曲线、5834资产核验
├── X3/                               # synthetic空跑/队列分解/外部160英语审计
└── X4/                               # LaTeX diff、claim、7表重放与复现候选
    ├── paper_source_R4/              # 实际Orange R4源树与10页PDF
    └── release_candidate_001/        # 数值CPU重放；尚未公开发布
artifacts/icde_eab_main_006/           # 各阶段manifest、原始审计、代码bundle
scripts/icde_eab_main_006/             # 全版本新脚本与失败修复，原脚本保留
```
''';C.prose(C.REPORT/'中文文件树.md',tree)
code=C.snapshot('final_001')
paths=[]
for folder in [C.ART,C.REPORT,C.ROOT/'scripts'/C.RUN,C.ROOT/'artifacts/icde_eab_main_006_logs_001']:
    for p in folder.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['manifest.json','hash_inventory.json']:paths.append(p)
    # Stage manifests/inventories also belong in the root inventory; only exclude root's future files.
    paths += [p for p in folder.rglob('*') if p.is_file() and p.name in ['manifest.json','hash_inventory.json'] and p.parent not in [C.ART,C.REPORT]]
paths.append(C.ROOT/'artifacts/icde_eab_main_006_closure_logs_001/root_close.log')
files={str(p.relative_to(C.ROOT)):C.sha(p) for p in sorted(set(paths))}
C.write(C.ART/'hash_inventory.json',{'files':files,'scope':'all closed run reports/artifacts/source/logs; external immutable model/index bytes verified in 5834-row asset ledger','active_writers':0})
main={'run_id':C.RUN,'status':'CPU_WORK_COMPLETED_GPU_BLOCKED_EAB_NOT_READY','time_bj':C.stamp(),'stages':records,'canonical_X2':'X2/rebuild_002','setup_supplement':'X2/cost_asset_supplement_003','unique_Dev_queries':164,'candidate_Dev_families':122,'Fit_queries':81,'new_GPU_quality_queries':0,'new_GPU_pilot':'BLOCKED_GPU_BUSY','new_AA':'NOT_RUN_GPU_BUSY','new_external_English_audit_queries':160,'new_external_retrieval_evaluation':0,'synthetic_mock_rows':730,'synthetic_mock_not_scientific_samples':True,'Cal_Eval_oldBL_test_CS_reads':0,'historical_frozen005_changed':False,'new_certification':'NOT_CERTIFIED','independent_Eval':'NOT_RUN_SEALED','decision':'EAB_CONDITIONAL_NOT_READY','verified_old_model_index_assets':5834,'CPU_replay_hash_identical_tables':7,'paper_revision':4,'paper_pages':10,'paper_PDF_SHA256':'5ed1d6ce18196c1312e5a3d8a87ff619e6e1cf524477e43e4a2d0b7ee6b4be15','code_snapshot_commit':code['code_snapshot_commit'],'code_bundle_SHA256':code['bundle_SHA256'],'base_project_commit':code['base_project_commit'],'inventory_SHA256':C.sha(C.ART/'hash_inventory.json'),'bound_files':len(files),'attribution_issue_disclosed':True}
C.write(C.ART/'manifest.json',main);C.write(C.REPORT/'manifest.json',main);C.write(C.REPORT/'hash_inventory.json',{'artifact_root_inventory':str((C.ART/'hash_inventory.json').relative_to(C.ROOT)),'SHA256':C.sha(C.ART/'hash_inventory.json')})
print(json.dumps({**main,'manifest_SHA256':C.sha(C.ART/'manifest.json')},ensure_ascii=False),flush=True)
