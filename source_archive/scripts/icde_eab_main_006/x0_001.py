import common as C
from pathlib import Path
import json, shutil, subprocess
C.start('X0');checks=[]
def check(p,expected,kind):
    p=Path(p);p=p if p.is_absolute() else C.ROOT/p
    actual=C.sha(p);checks.append({'path':str(p.relative_to(C.ROOT)),'kind':kind,'expected':expected,'actual':actual,'status':'PASS' if expected==actual else 'FAIL'})
    if actual!=expected:raise RuntimeError('HASH MISMATCH '+str(p))
old=C.ROOT/'artifacts/icde_eab_validation_005'
check(old/'manifest.json','6eb0c9c380dc2eb55cfaa6301a4ae6ff3dbafc972591b55958fe5ed1efb2b28e','PREDECESSOR')
m=C.load(old/'manifest.json');check(old/'hash_inventory.json',m['inventory_SHA256'],'INVENTORY')
inv=C.load(old/'hash_inventory.json')
for i,(p,h) in enumerate(inv['files'].items()):
    check(p,h,'PREDECESSOR_FILE')
    if i%500==0:print(json.dumps({'verified':i,'total':len(inv['files']),'time_bj':C.stamp()}),flush=True)
check(old/'code_final_001/code_snapshot_001.bundle',m['code_bundle_SHA256'],'CODE_BUNDLE')
f=C.ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
for name,h in {'protocol.json':'ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41','asset_lock.json':'116b3a4a486df6049f05221f9cb4856ab5c808605bfa6659d4fca530a9848bbe','freeze_record.json':'f276af09e40cad07c6ad180fc3a8ee8250e6565518b1db865b0f81178a465004'}.items():check(f/name,h,'FROZEN005')
check('reports/pharma_fixedcorpus_split_audit_001/audit_001/split_manifest.json','a70d0eec64f9611b0371a0f74fcba389f194798fd7b27d5e8e141b71f495440e','SPLIT')
C.write(C.A/'status_before.json',C.load('manifests/current_experiment_status.json','STATUS_METADATA'))
for name in ['locked_models.json','domain_policy_config.json','frozen_catalog.json']:C.write(C.A/name,C.load(old/'A0'/name))
authority=Path(__file__).parent/'authority'
for p in authority.glob('*.json'):
    data=json.loads(p.read_text(encoding='utf8'));C.write(C.A/'authority'/p.name,data)
    if p.name=='orange16_R2.json':C.prose(C.R/'preregistration.md',data['content'])
catalog={'run_id':C.RUN,'status':'FROZEN_EXECUTION_DESIGN_NO_UNSEAL','time_bj':C.stamp(),'epsilon':.05,'delta':.01,'primary_cost':'measured complete warm wall E2E; GPU/CPU separate','plans':C.PLANS,'policies':['Static','ideal static randomized envelope','Q_regret frozen','QP_regret frozen','RR_style_hard_adapted frozen available ablation','RR_style_soft_adapted frozen available ablation'],'oracle':'same catalog offline only','retrievers':{'primary':'vidore/colqwen2-v1.0-merged@2c9a09bb37ed19b63eb94ae6c29bb4e76eb6e3c2','second':'different-family trained late interaction, metadata/license audit before download; HR and Industrial 12 deterministic Dev queries each before outcomes'},'uncertainty':'2000 paired family/process bootstrap; pointwise descriptive','heldout':'SEALED; synthetic mock only','new_english_benchmark':'biomedical lectures V2 audit; preregister before any evaluation','AA':'unchanged 98.75% CI inside [0.9,1.1] all service/total 2/5QPS gates per domain; no lucky reruns','scope':'FIXED_CORPUS_QUERY_GENERALIZATION_DEV_EXPLORATORY'}
C.prose(C.R/'policy_catalog.yaml',json.dumps(catalog,ensure_ascii=False,indent=2));C.write(C.A/'frozen_execution_config.json',catalog)
guard={'default':'deny unknown role/project dataset paths in guarded acquisition wrappers','allow':['old HR/Finance/Industrial Dev119','Pharma Fit81 Dev45 role-filtered assets','unlabelled fixed corpus/index/model assets','prior authorized Dev reports and immutable code/manifests','official public model and benchmark metadata'],'deny':['Pharma Cal/Eval text labels outcomes','old raw B-L/test/CS','mixed Pharma qrels parquet','unregistered new-domain evaluation outcomes'],'implementation':'explicit input wrappers + CPython open audit rejection; not an OS-wide attestation','future_release':'separate explicit user approval required; no flag alone authorizes','counts':{'Pharma_Cal':0,'Pharma_Eval':0,'old_BL':0,'test':0,'CS':0}}
C.prose(C.R/'data_access_guard.yaml',json.dumps(guard,ensure_ascii=False,indent=2))
from data_guard import Guard
g=Guard(C.A/'guard_smoke.jsonl')
for p in ['pharma_eval_qrels.json','pharma_cal_outcomes.json','B-L_qrels.json','final_test_queries.json','unregistered_new_eval.csv']:
    try:g.read(C.ROOT/'data'/p,'EVAL');raise AssertionError('access should fail')
    except PermissionError:pass
C.write(C.A/'guard_smoke_result.json',{'denied_before_open':5,'sealed_payload_reads':0,'status':'PASS'})
matrix=[{'dataset':d,'queries':n,'families':fam,'split':'Dev','retriever':retr,'action_policy':p,'status':'NOT_RUN_PLANNED','evidence_level':'DEV_EXPLORATORY'} for d,n,fam in zip(C.DOM,[40,39,40,45],[24,25,29,44]) for retr in ['primary','secondary_if_pilot_gate_passes'] for p in C.PLANS+['RandomMix','Q_regret','QP_regret','RR-style adapted hard','RR-style adapted soft','Oracle_same_catalog']]
C.table(C.R/'experiment_matrix.csv',matrix);C.table(C.R/'integrity_checks.csv',checks)
C.write(C.A/'runtime_environment.json',{'python':sys.version if False else subprocess.check_output([str(C.ROOT/'.venv/bin/python'),'--version'],text=True).strip(),'GPU_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'storage':shutil.disk_usage(C.ROOT)._asdict(),'minimum_free_bytes':15*1024**3})
facts=f'PASS {len(inv["files"])}个005封账文件及bundle/冻结005/Pharma split逐项SHA一致；164 Dev/122候选族、Fit81，5项拒绝访问smoke通过。新实验矩阵80行仅计划，未填新成绩；Orange15实际R2（16正文仍写R1）已采用最新版本。'
C.prose(C.R/'X0_decision.md','# PASS_INPUTS_AND_EXECUTION_DESIGN_LOCKED\n\n'+facts+'\n\n先独立上报X0；随后X1审计、X2 CPU和X4论文并行。不得读取封存标签，第二模型未通过pilot前不扩展。\n')
C.finish('X0','PASS_INPUTS_AND_EXECUTION_DESIGN_LOCKED',facts,'开始X1预审/X2重建/X4写作；GPU忙则保留BLOCKED，不以CPU准备冒充GPU结果。')
