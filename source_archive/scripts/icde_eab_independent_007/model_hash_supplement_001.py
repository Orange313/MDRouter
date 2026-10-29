import common as C
C.start('T0/model_hash_supplement_001');rows=C.load('reports/icde_eab_main_006/X1/model_asset_checks.csv','PINNED_COLPALI_MODEL_HASH_LEDGER');checks=[];prep=C.load('artifacts/icde_eab_main_006/X1/preparation.json','PINNED_MODEL_PATHS')
for r in rows:
    path=__import__('pathlib').Path(prep['model_path'][r['repository']])/r['file']
    expected=r['actual_SHA256'];actual=C.sha(path);assert expected==actual,r
    checks.append({**r,'rechecked_SHA256':actual,'recheck_status':'PASS'})
C.table(C.R/'colpali_assets_rechecked.csv',checks);C.finish('T0_model_hash_supplement_001','PASS_COLPALI_PINNED_ASSETS_RECHECKED',f'{len(checks)}个已审计第二检索器资产SHA一致；权重未更改，私有key映射逐张量验证。','作为T0补充登记，不改动已封账T0或模型缓存。')
