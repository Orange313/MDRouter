import common as C
from pathlib import Path
import shutil,json,subprocess
C.start('L1_preparation_001')
old=C.ROOT/'artifacts/icde_eab_independent_007'
assert C.sha(old/'manifest.json')=='5c7f55e8745b58c1f04f1bb120d253cccfe2889b76c0ec010b262ab943bea22d'
m=C.load(old/'manifest.json','007_CLOSED_MANIFEST');assert C.sha(old/'hash_inventory.json')==m['inventory_SHA256']
inv=C.load(old/'hash_inventory.json','007_CLOSED_HASH_LEDGER')
oldp=old/'T1/biomedical_frozen_protocol.json';bio=C.load(oldp,'007_BENCHMARK_INPUT_LOCK');assert C.sha(oldp)==m['frozen_biomedical_protocol_SHA256']
for k,x in bio['official_inputs'].items():assert C.sha(C.ROOT/x['path'])==x['sha256']
pages=old/'T1_preparation/ocr_001/pages.json';assert C.sha(pages)==bio['OCR']['pages_SHA256']
assert shutil.disk_usage(C.ROOT).free>15*2**30
assets=C.load('reports/icde_eab_main_006/X1/model_asset_checks.csv','PINNED_MODEL_ASSETS');prep=C.load('artifacts/icde_eab_main_006/X1/preparation.json','PINNED_MODEL_PATHS');checks=[]
for x in assets:
    p=Path(prep['model_path'][x['repository']])/x['file'];actual=C.sha(p);assert actual==x['actual_SHA256'];checks.append({**x,'recheck_SHA256':actual,'status':'PASS'})
C.table(C.R/'colpali_asset_recheck.csv',checks)
protocol={'run_id':C.RUN,'time_bj':C.stamp(),'status':'DESIGN_RECORDED_BEFORE_NEW_COLPALI_OUTCOMES_BUT_AFTER_COLQWEN_RESULTS','evidence':'EXPLORATORY_POSTHOC_EXTENDED','queries':bio['query_ids'],'query_families':bio['query_families'],'sources':bio['source_groups'],'query_family_map':bio['query_family_map'],'corpus_pages':1016,'official_inputs':bio['official_inputs'],'same_OCR_SHA256':C.sha(pages),'models':{'adapter':'vidore/colpali-v1.3@1b5c8929330df1a66de441a9b5409a878f0de5b0','base':'vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a'},'plans':C.PLANS,'reference':'own ColPali Full-Visual exact valid-token MaxSim; masked true padding; negative maxima retained; stable score-desc/numeric-page-ID tie','risk':'r=max(U0-Uplan,0), nDCG@10 linear gain; r_bar and signed gap separate','epsilon':.05,'delta':.01,'new_K1':'NOT_CERTIFIED','training':'NONE; do not transfer Q/QP to changed reference','seed':C.SEED,'replication':'3 independent processes x3 shuffled timings/plan/query; separate GPU profiles;3warmups;1CPU torch thread; no Serving claim','statistics':'2000 pointwise paired bootstrap with27-source +157-family sensitivity and3processes; old/new retriever hardware cohorts not paired','copyright_empty_OCR_sensitivity':'all queries/pages retained; group queries by relevant empty-OCR page overlap; official removed-image extent unknown, not imputed','setup':'shared OCR measured007; new model/index/residency/BM setup separate; profile page-encoding GPU; missing cost never0','GPU':'3consecutive idle checks memory<128MiB/util0;100msResourceGuard,foreignPID abort own job','code_SHA256':{n:C.sha(Path(__file__).parent/n) for n in ['common.py','archive_code.py','colpali_loader_003.py','l1_biomedical_colpali_001.py']},'sealed_reads':0}
C.write(C.A/'protocol.json',protocol)
C.write(C.A/'preparation_complete.json',{'status':'PASS_PREPARATION_ONLY','time_bj':C.stamp(),'007_SHA256':C.sha(old/'manifest.json'),'protocol_SHA256':C.sha(C.A/'protocol.json'),'model_assets':len(checks),'disk_free_bytes':shutil.disk_usage(C.ROOT).free,'GPU_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)})
print(json.dumps({'status':'PASS_INPUTS_AND_L1_POSTHOC_DESIGN','protocol_SHA256':C.sha(C.A/'protocol.json'),'time_bj':C.stamp()}))
