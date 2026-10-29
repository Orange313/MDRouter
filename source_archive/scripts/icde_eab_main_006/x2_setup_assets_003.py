import common as C
from pathlib import Path
import json
C.A=C.ART/'X2/cost_asset_supplement_003';C.R=C.REPORT/'X2/cost_asset_supplement_003';C.A.mkdir(parents=True,exist_ok=False);C.R.mkdir(parents=True,exist_ok=False)
models=C.load('manifests/hr_models_verified.json');checks=[]
for name,m in models.items():
    if not isinstance(m,dict):continue
    for x in m.get('files',[]):
        p=Path(x['path']);h=C.sha(p);checks.append({'asset':str(p),'role':'MODEL_'+name,'expected':x['sha256'],'actual':h,'status':'PASS' if h==x['sha256'] else 'FAIL'});assert h==x['sha256']
indexes=['hr_primary_001','finance_primary_001','industrial_primary_003','pharma_fixedcorpus_index_001'];setup=[]
for d,index in zip(C.DOM,indexes):
    p=C.ROOT/'cache/indexes'/index;m=C.load(p/'index_manifest.json','UNLABELLED_INDEX_METADATA')
    for rel,h in m['files'].items():
        path=p/rel;actual=C.sha(path);checks.append({'asset':str(path.relative_to(C.ROOT)),'role':'UNLABELLED_INDEX','expected':h,'actual':actual,'status':'PASS' if actual==h else 'FAIL'});assert actual==h
    setup.append({'dataset':d,'index':index,'pages':m['pages'],'total_shared_index_setup_s':m.get('total_setup_seconds',m.get('total_setup_wall_s')),'visual_page_encoding_wall_s':m.get('visual_seconds'),'setup_GPU_active_s':m.get('visual_GPU_activity_s'),'setup_CPU_s':m.get('setup_CPU_s'),'peak_GPU_allocated_bytes':m.get('peak_GPU_allocated_bytes'),'setup_scope':'historical complete catalog index includes dense+visual for old three; Pharma visual-only. Not plan-marginal setup; not newly timed.','index_manifest_SHA256':C.sha(p/'index_manifest.json')})
    print(json.dumps({'dataset':d,'assets_so_far':len(checks),'time_bj':C.stamp()}),flush=True)
main=C.load(C.REPORT/'X2/rebuild_002/eab_main_table_dev.csv')
for r in main:
    s=next(x for x in setup if x['dataset']==r['dataset']);r['index_setup_s']=s['total_shared_index_setup_s'];r['index_setup_scope']=s['setup_scope'];r['marginal_plan_setup_s']=None;r['setup_GPU_active_s']=s['setup_GPU_active_s'];r['setup_CPU_s']=s['setup_CPU_s']
C.table(C.R/'eab_main_table_dev_with_setup.csv',main);C.table(C.R/'index_setup_inventory.csv',setup);C.table(C.R/'model_and_index_asset_recheck.csv',checks)
curves=[]
for r in main:
    ref=next(v for v in main if v['dataset']==r['dataset'] and v['action_policy']=='Full-Visual');s=float(r['index_setup_s']);a=float(r['wall_E2E_mean_ms'])/1000;b=float(ref['wall_E2E_mean_ms'])/1000
    for q in [100,1000,10000,100000,1000000]:curves.append({'dataset':r['dataset'],'policy':r['action_policy'],'Q_scenario':q,'average_wall_plus_equal_setup_s':a+s/q,'gain_vs_FullV_including_equal_setup_pct':100*(b-a)/(b+s/q),'assumption':'same already-built shared catalog, setup identical denominator; not marginal business cost','cost_evidence':r['cost_scope']})
C.table(C.R/'shared_setup_amortization_scenarios.csv',curves)
C.prose(C.R/'setup_scope.md','# PASS 历史setup补齐 / PASS模型索引重核\n\n主表的index_setup_s现在明确为既有共享catalog的初始构建wall，不是各计划增量建索引。前三域包括原dense构建，Pharma仅视觉；setup GPU与CPU仅有实测字段才填。BM25独用时可以不建视觉索引，但本轮没有同口径的独立marginal setup，不能把共享全量值当作BM25所需成本，也不填0。Q曲线仅给相同共享setup场景，不是业务净收益，Q未由用户冻结。\n\n本轮重新逐资产SHA，未读取任何Cal/Eval或旧B-L/test/CS标签。\n')
C.write(C.A/'result.json',{'status':'PASS_SETUP_AND_ASSET_RECHECK','verified_assets':len(checks),'time_bj':C.stamp(),'primary_model_and_four_indexes':'PASS','sealed_reads':0})
print(json.dumps({'status':'PASS_SETUP_AND_ASSET_RECHECK','verified_assets':len(checks),'time_bj':C.stamp()}),flush=True)
