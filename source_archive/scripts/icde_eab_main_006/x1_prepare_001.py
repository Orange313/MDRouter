import common as C
from pathlib import Path
import json, hashlib, shutil, subprocess, importlib.metadata as im
C.start('X1');S=Path(__file__).parent
for folder in ['official_metadata_001','official_sources_002','official_sources_003']:
    shutil.copytree(S/folder,C.A/folder)
pin=[('vidore/colpali-v1.3','1b5c8929330df1a66de441a9b5409a878f0de5b0','colpali_pinned_api'),('vidore/colpaligemma-3b-pt-448-base','30ab955d073de4a91dc5a288e8c97226647e3e5a','base_pinned_api')]
checks=[];local={}
for repo,revision,source in pin:
    api=json.loads((S/'official_sources_003'/(source+'.payload')).read_bytes());assert api['sha']==revision
    path=Path('/path/to/user/.cache/huggingface/hub')/('models--'+repo.replace('/','--'))/'snapshots'/revision
    assert path.is_dir();local[repo]=str(path)
    expected={x['rfilename']:x.get('lfs',{}).get('sha256') for x in api['siblings']}
    for p in path.iterdir():
        if not p.is_file():continue
        h=C.sha(p);e=expected.get(p.name)
        checks.append(dict(repository=repo,revision=revision,file=p.name,bytes=p.stat().st_size,official_LFS_SHA256=e,actual_SHA256=h,status='PASS_OFFICIAL_LFS' if e==h else ('PASS_LOCAL_PIN_NO_OFFICIAL_SHA256' if e is None else 'FAIL')))
        if e is not None:assert h==e
sources=C.load('manifests/primary_datasets_inventory.json')
sample=[]
for d in [C.DOM[0],C.DOM[2]]:
    a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz')
    pairs=sorted(zip(a['qids'].tolist(),a['family_primary'].tolist()),key=lambda x:hashlib.sha256((str(C.SEED)+d+str(x[1])+str(x[0])).encode()).hexdigest())
    seen=set()
    for q,g in pairs:
        if g in seen:continue
        sample.append(dict(domain=d,query_id=int(q),family=str(g),selection='seeded hash, unique family; no outcomes used'));seen.add(g)
        if len(seen)==12:break
C.table(C.R/'pilot_query_allowlist.csv',sample);C.table(C.R/'model_asset_checks.csv',checks)
provenance={'status':'TRAINED_ADAPTER_PLUS_BASE_VERIFIED_NO_NEW_TRAINING','current_reference':{'repo':'vidore/colqwen2-v1.0-merged','revision':'2c9a09bb37ed19b63eb94ae6c29bb4e76eb6e3c2','family':'Qwen2-VL'},'secondary':{'adapter_repo':pin[0][0],'adapter_revision':pin[0][1],'base_repo':pin[1][0],'base_revision':pin[1][1],'family':'PaliGemma/SigLIP/Gemma','trained_checkpoint':True,'adapter_license':'MIT','base_license':'Gemma terms, research use; weights not redistributed','local_paths_private_only':local,'training_data':'127460 query-page pairs academic train + synthetic web PDFs; page-level no-overlap with project Dev NOT_PROVEN','processor':'ColPaliProcessor from pinned adapter; native448 square page image, no extra OCR input','scoring':'FP16 embeddings converted FP32 exact MaxSim, zero padding/clamp_min(0); stable (-score,numeric corpusID)','engine_version':im.version('colpali-engine'),'transformers_version':im.version('transformers')},'pilot':{'domains':['HR','Industrial'],'queries_per_domain':12,'corpus_pages':[1110,5244],'timing_processes':3,'repeats_per_query_plan':3,'warmup_per_plan':3,'plans':['BM25-R50','BM25-R100','Full-Visual'],'GPU_profiles':'separate from primary unprofiled wall','scorer_correctness':'exact batch scorer vs official processor and per-page MaxSim; candidate containment implies same Top10 including tie-break','gate':'finite embeddings, official scorer and stable ranking agree within FP32 tolerance; exact output consistent across processes; uncontaminated resource guard; storage>=15GiB. No quality gain gate or model cherry-picking','expand':'only after pilot gate & resource budget; current stage stops at small preregistered pilot if insufficient'},'old_Q_QP_new_reference':'INCOMPATIBLE_NOT_RETRAINED'}
C.prose(C.R/'model_provenance.yaml',json.dumps(provenance,ensure_ascii=False,indent=2));C.write(C.A/'pilot_protocol.json',provenance)
estimate=6354*1056*128*2
C.prose(C.R/'gpu_memory_and_index_budget.md',f'# 资源预算（预登记估计，非实测）\n\n现有完整基座+adapter缓存可复用，官方LFS SHA已核验。HR1110页+Industrial5244页；FP16索引约{estimate/2**30:.2f}GiB（按1056token/page估计），每域单独常驻FP32检索。模型约6GiB，Industrial常驻索引约2.65GiB，含激活保守显存预算<20GiB；A6000 48GiB。实际峰值须测，不能称估计为实测。索引不扩容、不删历史；每批检查剩余>=15GiB。预计10–40分钟索引，硬上限由资源/实测决定。\n\n仅官方已训练adapter + 固定基座，不用未训练base充当检索器。MIT adapter不覆盖Gemma基座条款；公开artifact只提供获取脚本/revision/hash，不转发权重。训练重叠仍UNKNOWN，因此只支持Dev跨模型机制，不支持独立性主张。\n')
C.write(C.A/'preparation.json',{'time_bj':C.stamp(),'model_assets':len(checks),'official_LFS_verified':sum(x['status']=='PASS_OFFICIAL_LFS' for x in checks),'locked_queries':len(sample),'sample_SHA256':C.sha(C.R/'pilot_query_allowlist.csv'),'model_path':local,'disk_free_bytes':shutil.disk_usage(C.ROOT).free,'GPU':'NOT_RUN','GPU_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)})
print(json.dumps({'status':'PASS_CPU_AUDIT_GPU_NOT_RUN','time_bj':C.stamp(),'model_assets':len(checks),'queries':len(sample)}),flush=True)
