import common as C
import importlib.metadata as M,subprocess,sys,platform,requests
C.start('T5/preparation_001')
deps={}
for name in ['torch','transformers','colpali-engine','peft','safetensors','huggingface-hub','numpy','scipy','pandas','pyarrow','Pillow','requests']:
    try:deps[name]=M.version(name)
    except M.PackageNotFoundError:deps[name]='MISSING'
C.write(C.A/'environment.json',{'time_bj':C.stamp(),'python':sys.version,'platform':platform.platform(),'versions':deps,'GPU_driver':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader'],text=True),'shared_venv_modified':False})
urls=['https://icde2027.github.io/submission-guidelines.html','https://icde2027.github.io/cf-research-papers.html'];sources=[]
for i,url in enumerate(urls):
    r=requests.get(url,timeout=60);r.raise_for_status();p=C.A/f'icde_guidelines_{i}.html';C.prose(p,r.text);sources.append({'url':url,'time_bj':C.stamp(),'sha256':C.sha(p)})
C.write(C.A/'official_guidelines_receipt.json',sources)
readme=r'''# MDRouter EAB artifact preparation — icde_eab_independent_007

This is a private reproducibility preparation, not a public artifact release or acceptance claim.

## Data and evidence

* HR40 / Finance39 / Industrial40 / Pharma Dev45 are repeatedly exposed Dev data. The frozen Q/QP and adapted hard/soft checkpoints were trained on allowed Pharma Fit81, not Biomedical.
* Biomedical English: `vidore/biomedical_lectures_eng_v2@e82eb2f3e9f701688e6f3ba41fffad68c5b3b5fa`, 160 queries / 1016 pages / 27 sources / 515 qrels. CC-BY-NC-SA-4.0; preserve attribution and acquire from the official repository. Some released slides remove copyrighted images. No raw-page redistribution is included in a public bundle.
* Pretraining contamination remains UNKNOWN. This benchmark was unused for MDRouter model selection; that is a narrower claim than complete pretraining independence.
* ColPali adapter `1b5c8929330df1a66de441a9b5409a878f0de5b0`, base `30ab955d073de4a91dc5a288e8c97226647e3e5a`. MIT adapter and Gemma base terms apply separately. Primary ColQwen2 revision `2c9a09bb37ed19b63eb94ae6c29bb4e76eb6e3c2`; retain pinned model-card terms. Do not redistribute weights.
* Pharma Cal81/Eval73 and old B-L/test/CS remain sealed. Acquisition/replay must use only approved role-filtered Fit/Dev assets and official new Biomedical files. Do not execute a mixed Pharma qrels loader.

## Environment

Use the version lock in `environment.json`. Shared `.venv` was not modified. Tesseract4.1.1 and its runtime packages were extracted into a project-private directory; its English OCR uses PSM3, four CPU workers, with package/file hashes in the OCR runtime lock. Retain20 empty-OCR pages.

## Execution order

1. Rehash prior manifests/assets and freeze an exclusive new run directory.
2. Audit new data and freeze all query IDs, 157 query-family groups and 27 source groups before quality evaluation.
3. Run `t3_colpali_pilot_004.py` / HR retry005 on frozen24 queries; retain failed001/002 and GPU-busy004 attempts. Correct MaxSim masks padding to minus infinity and retains negative maxima. Tolerance1e-3 was not loosened; historical005 is unchanged.
4. Execute `t2_biomedical_exact_004.py GPU PROCESS` for3 independent processes, 3 requests/method/query and seeded shuffled order. Static FullV/R50/R100/BM25, frozen Q/QP, fixed RandomMix and adapted hard/soft all execute actual complete paths. GPU profiles are a separate cohort.
5. After Biomedical reporting, run the four legal Dev-domain `t3_colpali_expansion_004.py GPU DOMAIN PROCESS` paths only if resource admission passes.
6. Summarize via `t2_summarize_001.py`, `t3_pilot_summary_005.py`, optional `t3_expansion_summary_004.py`, and `t4_assemble_001.py`. These scripts use exclusive outputs; rerun under a new ID/output location, never overwrite prior results.

## CPU numeric replay and figures

After the run completes, with immutable bound inputs present:

    .venv/bin/python scripts/icde_eab_independent_007/replay_numeric_tables_001.py --output artifacts/icde_eab_independent_007/T5/replay_001

This recomputes report CSVs and three PDF/PNG figures from recorded per-query quality, raw unprofiled timings, and separate profiles. Compare every numeric CSV hash. It does not create new quality observations, independent samples or GPU measurements. PDF container timestamps can differ; numeric input hashes and rendered content are the relevant checks.

## Statistical and cost interpretation

2,000 pointwise paired cluster/process bootstrap repetitions; query-family and27-source sensitivity are both reported. They are descriptive intervals, not IID-production guarantees or new K1 certificates. Benchmark-fitted envelopes and gold allocations are retrospective/offline bounds. Frozen-policy actual risk/cost are primary; never select a new threshold from Biomedical results.

Report E2E mean/P50/P95, process and thread CPU, measured GPU active time, probe/Q/planner costs, OCR/index/model load and setup/Q assumptions separately. Missing costs stay missing. Warm resident-path savings are neither full business profit nor Serving throughput improvements. The prior strict A/A gate remains insufficient.

## Public-release gate

The ICDE2027 official guidelines require EAB reproduction artifacts and limit the main paper to12 pages excluding references and AI-content acknowledgment, with no appendix. Author/ORCID information, license-compliant public acquisition recipes, accessibility and clean-checkout replay must be checked before submission. Do not publish sealed data, private Orange authority snapshots, credentials, raw model weights or copyrighted page screenshots. This preparation does not itself authorize external publication.
'''
C.prose(C.R/'artifact_readme_draft.md',readme)
print({'status':'PASS_PREPARATION_ONLY_NOT_PAPER_COMPLETE','path':str(C.R),'time_bj':C.stamp()})
