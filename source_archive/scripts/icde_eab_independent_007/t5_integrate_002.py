"""Close manuscript/artifact preparation using actual immutable results."""
import common as C
import json,shutil,subprocess,csv,sys
from pathlib import Path
C.start('T5/integration_002')
incoming=C.ROOT/'artifacts/icde_eab_independent_007_paper_transfer_001'
qa=C.load(incoming/'pdf_qa.json','ACTUAL_PDF_VISUAL_QA');receipt=C.load(incoming/'receipt.json','ORANGE_NATIVE_PAPER_BUILD')
assert qa['status']=='PASS_RENDER_AND_NUMERIC_SOURCE_QA' and qa['visual_pages_inspected']==list(range(1,12))
assert C.sha(incoming/'paper_R7.pdf')==qa['PDF_SHA256']==receipt['pdf']['contentHash']
assert C.sha(incoming/'paper_source_R7.zip')==qa['source_bundle_SHA256']
assert receipt['build']['status']=='succeeded' and receipt['revision']==7
for name in ['paper_R7.pdf','paper_source_R7.zip','receipt.json','pdf_qa.json']:
    shutil.copyfile(incoming/name,C.A/name)
    if name.endswith('.pdf'):shutil.copyfile(incoming/name,C.R/name)
replay=C.load(C.ART/'T5/replay_003/replay_result.json','ACTUAL_CPU_NUMERIC_REPLAY')
assert replay['status']=='PASS' and len(replay['checks'])==15 and all(x['identical'] for x in replay['checks'])
proto=C.load(C.ART/'T1/biomedical_frozen_protocol.json','FROZEN_BEFORE_EVALUATION_PROTOCOL')
env=C.load(C.ART/'T5/preparation_001/environment.json','MEASURED_ENVIRONMENT_LOCK');C.write(C.R/'environment.lock.json',env)
requirements='\n'.join(f'{k}=={v}' for k,v in env['versions'].items() if v!='MISSING')+'\n'
C.prose(C.R/'requirements-observed.txt',requirements)
lock={'status':'PINNED_OFFICIAL_RECIPE_NOT_PUBLIC_RELEASE','biomedical_files':[],
      'models':[{'repo':'vidore/colqwen2-v1.0-merged','revision':proto['primary_model_revision'],'license':'Use official pinned model card; no weight redistribution'},
                {'repo':'vidore/colpali-v1.3','revision':'1b5c8929330df1a66de441a9b5409a878f0de5b0','license':'MIT adapter; separate base terms apply'},
                {'repo':'vidore/colpaligemma-3b-pt-448-base','revision':'30ab955d073de4a91dc5a288e8c97226647e3e5a','license':'Gemma base terms; official access/license acknowledgment required'}],
      'excluded':'No mixed historical qrels, Cal/Eval, old B-L/test/CS, credentials, private Orange authority or weights in a public package.'}
for role,x in proto['official_inputs'].items():
    repo_path=role+'/'+Path(x['path']).name
    lock['biomedical_files'].append({'role':role,'project_path':x['path'],'repo_path':repo_path,'sha256':x['sha256'],'url':f'https://huggingface.co/datasets/{proto["benchmark"]}/resolve/{proto["revision"]}/{repo_path}','license':'CC-BY-NC-SA-4.0; retain attribution, official notices and removed-image limitations'})
C.write(C.R/'public_acquisition_lock.json',lock)
commands=[sys.executable,str(C.ROOT/'scripts'/C.RUN/'acquire_public_assets_001.py'),'verify-existing','--lock',str(C.R/'public_acquisition_lock.json'),'--root',str(C.ROOT)]
v=subprocess.run(commands,text=True,capture_output=True);C.prose(C.R/'public_input_hash_verification.log',v.stdout+v.stderr);assert v.returncode==0
sources={'ColPali_pilot_HR':'t3_colpali_pilot_005.py','ColPali_pilot_Industrial':'t3_colpali_pilot_004.py','Biomedical_index_and_full_execution':'t2_biomedical_exact_004.py','ColPali_expansion_old3':'t3_colpali_expansion_004.py','ColPali_expansion_Pharma':'t3_colpali_expansion_006.py','ColPali_summary':'t3_expansion_summary_007.py','Biomedical_summary':'t2_summarize_001.py','T4_tables_figures':'t4_assemble_004.py','CPU_replay':'replay_numeric_tables_003.py','primary_parity':'primary_scorer_parity_002.py','official_acquisition':'acquire_public_assets_001.py'}
C.write(C.R/'approved_entrypoints.json',{k:{'script':v,'sha256':C.sha(C.ROOT/'scripts'/C.RUN/v)} for k,v in sources.items()})
assets=[]
for index in sorted((C.ROOT/'cache/indexes').glob(C.RUN+'*')):
    p=index/'index_manifest.json'
    if p.exists():
        x=C.load(p,'UNLABELLED_INDEX_MANIFEST');assets.append({'index':str(index.relative_to(C.ROOT)),'index_manifest_SHA256':C.sha(p),'pages':x.get('pages',x.get('corpus_pages')),'setup_wall_s':x.get('setup_wall_s'),'setup_CPU_s':x.get('setup_CPU_s'),'setup_GPU_s':x.get('setup_GPU_activity_s'),'GPU_missing_reason':x.get('setup_GPU_scope',''),'peak_GPU_bytes':x.get('peak_GPU_bytes',x.get('peak_GPU_allocated_bytes'))})
C.table(C.R/'new_index_asset_ledger.csv',assets)
readme='''# MDRouter 007 reproducibility preparation

This is an executed, private workspace artifact, not a public artifact release. No new model was trained. The authoritative native Paper is R7, build `{build}`, PDF SHA256 `{pdf}`. It has 11 pages including references. Eight agreed sections and prior user prose are retained. Authors, affiliations, ORCIDs and the final AI-use acknowledgment remain pending.

## Evidence and licenses

Biomedical: `vidore/biomedical_lectures_eng_v2@e82eb2f3e9f701688e6f3ba41fffad68c5b3b5fa`, all160 English queries /157 conservative families /27 sources /1016 pages /515 graded judgments. Official questions are generated and human-reviewed, not production IID requests. Public slides remove some copyrighted illustrations. CC-BY-NC-SA-4.0: use official acquisition, retain attribution and all notices; do not include raw slide images in a public bundle. Model pretraining overlap remains UNKNOWN. The scope is unused in MDRouter model selection, not proven unseen to every pretrained model.

The old HR40/Finance39/Industrial40/Pharma Dev45 are exposed development evidence. Frozen Q/QP and same-capacity hard/soft controls were fitted on allowed Pharma Fit81. Use only archived role-filtered projections. No recipe downloads the old mixed qrels to reconstruct Cal/Eval, B-L, test or CS.

The model pins and official acquisition links are in `public_acquisition_lock.json`. MIT ColPali adapter and Gemma base terms apply separately. Primary model terms come from its pinned official card. Acquire weights from official sources; no weight redistribution is authorized by this artifact. Full consumed-model/index hash ledgers are linked through T0 and T3; `new_index_asset_ledger.csv` lists new indexes. Software license review for a public release remains pending.

## Environment and source

Python3.11.15, A6000 driver570.133.20 and measured library versions are in `environment.lock.json`; `requirements-observed.txt` records versions, not a promise that all future package repositories still contain identical wheels. The shared `.venv` was unchanged. Tesseract4.1.1 was privately extracted, PSM3/English/four CPU workers; use its original runtime-lock and package hashes. Keep all20 empty OCR pages.

Each stage has a private Git source bundle and commit. Clone the final `artifacts/icde_eab_independent_007/code_final_001/source.bundle` to inspect versioned source; it includes retained failed/deprecated scripts. Run only `approved_entrypoints.json`, not an arbitrary older numeric suffix. The current numerical replay is workspace-bound: `common.ROOT` points at the recorded project. A clean public checkout with path-independent data mounting has NOT been validated and is a submission blocker. Do not call this portable reproduction.

## One-command verified numeric replay

From the recorded project root, choose a new, nonexistent output directory:

    .venv/bin/python scripts/icde_eab_independent_007/replay_numeric_tables_003.py --output /absolute/new/replay_output

This reconstructs 15 numeric CSVs and three figure families from immutable raw quality, actual unprofiled request logs and separately profiled GPU traces. The completed `T5/replay_003/replay_result.json` is PASS with every CSV hash identical. It creates no new independent queries, GPU measurements or certificates. PDF timestamps may differ; compare numeric CSV hashes and render content.

## Acquisition and GPU/index commands

    .venv/bin/python scripts/icde_eab_independent_007/acquire_public_assets_001.py plan --lock reports/icde_eab_independent_007/T5/integration_002/public_acquisition_lock.json
    .venv/bin/python scripts/icde_eab_independent_007/acquire_public_assets_001.py verify-existing --lock reports/icde_eab_independent_007/T5/integration_002/public_acquisition_lock.json --root /path/to/project

The same entry offers `fetch-biomedical` and `fetch-models` into a fresh destination with explicit official-license acknowledgment. Never point it at an existing data/cache directory. Maintain at least15GiB free. Official new Biomedical files are the only labels fetched by the recipe.

The exact executed GPU workers are listed below. These historical commands must NOT be rerun in the finished namespace: exclusive-output checks deliberately refuse. A fresh isolated replay workspace/output namespace and matching locked inputs are required before a new hardware repeat. This run does not authorize another evaluation or overwrite.

    .venv/bin/python scripts/icde_eab_independent_007/t3_colpali_pilot_005.py GPU vidore_v3_hr PROCESS
    .venv/bin/python scripts/icde_eab_independent_007/t3_colpali_pilot_004.py GPU vidore_v3_industrial PROCESS
    .venv/bin/python scripts/icde_eab_independent_007/t2_biomedical_exact_004.py GPU PROCESS
    .venv/bin/python scripts/icde_eab_independent_007/t3_colpali_expansion_004.py GPU DOMAIN PROCESS
    .venv/bin/python scripts/icde_eab_independent_007/t3_colpali_expansion_006.py GPU Pharmaceuticals PROCESS

`PROCESS`=0,1,2; run process0 first to build each missing index from the pinned corpus, then processes1/2 reuse it. The expansion DOMAIN values are vidore_v3_hr, vidore_v3_finance_en, vidore_v3_industrial. Pharma uses006 because the archived role literal is `development`; failed004 attempts are retained. Canonical pilot HR005/Industrial004, Bio004, static expansion old3=004/Pharma006; summaries Bio001/pilot005/expansion007; final tables004. Actual end-to-end stage times are recorded in start/completion manifests and original logs, not a promised runtime on a different host. Bio produced12960 unprofiled timings; ColPali expansion5904; the pilot648. Timing repeats do not increase quality sample n.

## Statistical and cost contract

Pointwise paired 2000-replicate family/process intervals plus27-source sensitivity. This is descriptive conditional benchmark evidence, not production PAC risk, not a fresh K1. Benchmark-fitted static/random envelopes and gold allocations are retrospective bounds, not deployed plans or test-selected checkpoints. Do not refit from Biomedical outcomes.

Report actual U=nDCG@10, r=max(U0-Ua,0), r_bar and signed gap separately. CPU/process, GPU-active, wall mean/P50/P95, probes, feature extraction, planner and setup/Q are distinct. Do not add overlapping CPU/GPU/wall durations. New paths execute probes and record reuse; old adapted Dev rows use composed estimates. ColPali unprofiled setup GPU remains missing. No full energy, business net benefit or strict Serving superiority is established. All failed/busy attempts and negative points remain in the inventory.

## Submission gate

EAB analysis direction: CONDITIONAL_GO. Novel learned-router method/production guarantee: NOT_ESTABLISHED. Before submission: author/ORCID/AI-use completion, license-clean anonymous public artifact, clean-checkout replay, external accessibility and final editorial review. Strict Serving remains INSUFFICIENT; do not add a throughput-superiority claim. No new Cal/Eval or old B-L/test/CS reading is authorized. Official ICDE2027 Research/EAB guidelines:12 pages excluding references and AI-content acknowledgment, no appendix, reproduction artifacts required. Archived official HTML/URLs are under T5/preparation_001.
'''.format(build=receipt['build']['id'],pdf=qa['PDF_SHA256'])
C.prose(C.R/'artifact_readme.md',readme)
C.prose(C.R/'EAB_go_nogo.md','# EAB evidence gate\n\n**CONDITIONAL_GO_FOR_ANALYSIS_PAPER / NOT_SUBMISSION_READY.** The previously missing Biomedical160 execution and distinct ColPali-family measurements now exist. All domains, negative results, CPU/GPU/actual wall and uncertainty are retained. Learned-policy cost advantage remains unproven; probe-conditioned signal does not overcome the measured same-risk static/random envelope. New K1 is NOT_CERTIFIED and strict Serving is INSUFFICIENT.\n\nPublic artifact access/clean-checkout portability, source-license review, authors/ORCIDs and AI-use acknowledgment remain pending. These are explicit submission blockers, not reasons to invent a method benefit or open sealed data. No further model expansion is recommended from the present results.\n\nSupports paper claims F1/F2 with stated scope; F3 formal guarantee and F4 serving superiority remain pending/insufficient. See T4/rebuild_004/eab_claim_to_evidence_matrix.md.\n')
C.prose(C.R/'paper_update_report.md',f'# PASS native Paper R7\n\nActual build {receipt["build"]["id"]};11 pages;PDF SHA256 {qa["PDF_SHA256"]}. All11 pages visually inspected.13 LaTeX/bibliography/compiler input hashes match the downloaded source bundle. Bio9/ColPali16 rows were CSV-generated. R5/R6 intermediate history retained; R7 removes duplicated tail discovered by PDF QA. Fifteen numeric CSVs replay identical. The paper retains user Introduction/Related Work and eight-section structure. It explicitly distinguishes exposed Dev, preregistered descriptive benchmark, actual/composed costs, legacy/corrected scoring, post-hoc diagnostics and absent new certification/strict Serving guarantee.\n\nPublic release and submission metadata remain pending.\n')
C.finish('T5_integration_002','PASS_PAPER_AND_PRIVATE_REPLAY_INSUFFICIENT_PUBLIC_SUBMISSION','原生Paper R7实际编译、11页逐页核验、13输入hash一致；15数值CSV CPU重放SHA一致；数据/模型获取与索引执行入口、版本环境及失败记录已归档。新160 benchmark与164 exposed Dev严格分层。','EAB分析方向条件继续，学习路由收益未证明；公共工件/可移植clean-checkout/作者信息等未完成，不标投稿就绪。')
