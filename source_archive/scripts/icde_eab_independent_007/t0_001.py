import common as C
import subprocess,shutil,json,csv,sys
from pathlib import Path
C.start('T0');checks=[]
def check(p,h,role):
    p=Path(p);p=p if p.is_absolute() else C.ROOT/p
    a=C.sha(p);checks.append({'path':str(p),'role':role,'expected':h,'actual':a,'status':'PASS' if a==h else 'FAIL'})
    assert a==h,'HASH MISMATCH '+str(p)
old=C.ROOT/'artifacts/icde_eab_main_006'
check(old/'manifest.json','76d40d87985d5fa7b41b464da47cf9e88c16fe8f1e5b5e924e22f70f0309508f','006_FINAL')
m=C.load(old/'manifest.json','PRIOR_MANIFEST');check(old/'hash_inventory.json',m['inventory_SHA256'],'INVENTORY')
inv=C.load(old/'hash_inventory.json','PRIOR_HASH_LEDGER')
for i,(p,h) in enumerate(inv['files'].items()):
    check(p,h,'006_BOUND_FILE')
    if i%500==0:print(json.dumps({'verified_006':i,'total':len(inv['files']),'time_bj':C.stamp()}),flush=True)
check(old/'code_final_001/source.bundle',m['code_bundle_SHA256'],'006_CODE_BUNDLE')
assets=C.load('reports/icde_eab_main_006/X2/cost_asset_supplement_003/model_and_index_asset_recheck.csv','MODEL_INDEX_HASH_LEDGER')
for i,r in enumerate(assets):
    check(r['asset'],r['expected'],'MODEL_INDEX')
    if i%1000==0:print(json.dumps({'model_index_verified':i,'total':len(assets),'time_bj':C.stamp()}),flush=True)
freeze=C.ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
for name,h in {'protocol.json':'ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41','asset_lock.json':'116b3a4a486df6049f05221f9cb4856ab5c808605bfa6659d4fca530a9848bbe','freeze_record.json':'f276af09e40cad07c6ad180fc3a8ee8250e6565518b1db865b0f81178a465004'}.items():check(freeze/name,h,'FROZEN_005')
check('reports/pharma_fixedcorpus_split_audit_001/audit_001/split_manifest.json','a70d0eec64f9611b0371a0f74fcba389f194798fd7b27d5e8e141b71f495440e','PHARMA_SPLIT')
for n in ['locked_models.json','domain_policy_config.json','frozen_catalog.json']:C.write(C.A/n,C.load(old/'X0'/n,'FROZEN_POLICY'))
allow=C.load('reports/icde_eab_main_006/X1/pilot_query_allowlist.csv','FROZEN_24_PILOT_IDS');C.table(C.R/'colpali_pilot_allowlist.csv',allow)
for p in (Path(__file__).parent/'authority').iterdir():
    if p.is_file():shutil.copyfile(p,C.A/p.name)
env={'time_bj':C.stamp(),'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'disk':shutil.disk_usage(C.ROOT)._asdict(),'min_free_bytes':15*2**30,'python':sys.version,'shared_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip()}
assert env['disk']['free']>15*2**30;C.write(C.A/'environment_resource_snapshot.json',env)
config={'status':'FROZEN_EXECUTION_BOUNDARIES','time_bj':C.stamp(),'seed':C.SEED,'risk':'max(nDCG10_reference - nDCG10_action,0); linear gain; r_bar and signed gap separate','epsilon':.05,'delta':.01,'reference':'model-specific exact Full Visual stable score descending page-ID ascending','cost':'actual complete warm per-query path; CPU/GPU separately; setup/Q curves not full business benefit','GPU_order':['ColPali frozen 24-query pilot','Biomedical160 full main','ColPali expansion only after pilot PASS and Biomedical done'],'GPU_guard':'3 consecutive 1-second idle checks memory<128MiB utilization=0; ResourceGuard 100ms foreign PID abort own job; no killing others','pilot_replication':'3 independent processes x 3 requests per plan/query with seeded shuffled order; 3 warmups','data_boundaries':{'allowed':['old Dev119','Pharma Fit81/Dev45 role-filtered','unlabelled corpus/index/model','Biomedical official160 after T1 preregistration'],'sealed_read_counts':{'Pharma_Cal81':0,'Pharma_Eval73':0,'old_BL':0,'old_test':0,'CS':0}},'new_certification':'NOT_CERTIFIED','retrievers':{'primary':'vidore/colqwen2-v1.0-merged@2c9a09bb37ed19b63eb94ae6c29bb4e76eb6e3c2','second':'vidore/colpali-v1.3@1b5c8929330df1a66de441a9b5409a878f0de5b0 trained adapter + base@30ab955d073de4a91dc5a288e8c97226647e3e5a'},'Biomedical':'audit/preregister all160 before outcomes; pretraining UNKNOWN qualifies scope, unexplained actual contamination blocks','A_A':'no new Serving superiority absent strict equivalence gate','new_models':'none trained; frozen Q/QP/RR-adapted only'}
C.write(C.A/'frozen_execution_config.json',config)
C.table(C.R/'integrity_checks.csv',checks)
C.table(C.R/'experiment_matrix.csv',[{'dataset':d,'queries':n,'model':r,'path':p,'status':'NOT_RUN_PLANNED','evidence':e} for d,n,e in [('HR',40,'DEV_EXPLORATORY'),('Finance-EN',39,'DEV_EXPLORATORY'),('Industrial',40,'DEV_EXPLORATORY'),('Pharma',45,'DEV_EXPLORATORY'),('Biomedical',160,'PREREGISTER_BEFORE_EVALUATION')] for r in ['ColQwen2','ColPali_after_pilot'] for p in C.PLANS+['RandomMix','Q_regret','QP_regret','RR_adapted_hard','RR_adapted_soft','Oracle_offline']])
C.prose(C.R/'T0_decision.md','# PASS｜输入核验与边界冻结\n\n'+f'006封账 {len(inv["files"])} 文件、{len(assets)} 模型/索引资产、005协议与Pharma分区 hash 一致。GPU快照见环境日志，运行前再次三检；封存读取0。\n\n先上报Orange T0，再启动已冻结24查询 ColPali correctness pilot；T1完成前不计算Biomedical检索成绩。\n')
C.finish('T0','PASS_INPUT_INTEGRITY_AND_BOUNDARIES',f'{len(inv["files"])}个006封账文件、{len(assets)}模型/索引资产及005/split SHA全部一致；24条pilot IDs不变，封存读取0。','先独立追加Orange06，再执行ColPali pilot与Biomedical来源/协议审计。')
