# Release candidate010 — PRIVATE, PENDING OWNER APPROVAL

This merged candidate retains the007 base and009 cost addendum. It is not published, licensed as MIT, or a new scientific experiment. No models, page images, full qrels, vector indexes, or sealed partitions are distributed.

```sh
python -m venv .venv-replay
. .venv-replay/bin/activate
pip install -r requirements-replay.txt
python reproduce_all.py --output /absolute/path/NEW_REPLAY
```

Expected:38 numeric CSV hashes (30 base+8 cost), five figure families. Numerical replay does not repeat GPU timing. Unprofiled009 and old007/008 cohorts remain separate. See cost009/ and provenance_010.json. Upstream licenses remain separate; LICENSE_PROPOSED.md grants no rights.

# Multimodal Retrieval Physical-Plan Benchmark — release candidate

**UNPUBLISHED / PENDING OWNER REVIEW.** Do not treat this directory as a public anonymous artifact. The owner must approve the exact commit, file inventory and software license before any push. The public GitHub artifact will require no reviewer login after owner-approved publication. ICDE is single-blind; author anonymization is not an additional release requirement.

## Clean-checkout numerical reproduction

Python3.11 with `requirements-replay.txt`; no GPU/network/original project is needed for numeric replay:

    python reproduce.py --output outputs/replay_001

The exclusive output directory must not exist. This decompresses small approved outcome/timing snapshots, runs the original numerical and figure scripts and checks all30 CSV hashes (15 unchanged007 plus15 new008 diagnostics). Three figure families are regenerated as PDF/PNG. `reproduction_receipt.json` is the actual result; an unrun command is not a PASS. Native PDF byte hashes can differ because of metadata; numerical hashes and plotted data are compared. Repeated timings/bootstrap draws do not create independent queries.

## Evidence scope and raw records

Biomedical007:160 English queries,157 conservative families,27 lecture sources,1016 corpus pages;9 frozen methods,12960 complete warm-path timings and2080 separate GPU profiles. Q/QP/RR-adapted coefficients are small frozen linear models, not large pretrained weights. Old four-domain164 Dev rows are exposed development evidence; their RR-adapted rows have composed costs. This is not an official RetrievalRouter reproduction. The snapshots include original raw request rows, paired grouping, GPU activity summaries and resource guards; source hashes/provenance are bound in `input_manifest.json` and `source_map.json`. No raw sealed relevance data are included. The historical resident cohort is archived separately, not treated as new benchmark data.

The finalized ColPali×Biomedical008 extension (160 queries,5760 request timings,1600 separateGPU records) is included. It is post-hoc after ColQwen outcomes and uses its own FullV; no frozen policy is retuned. The main15-CSV reproduction is the unchanged007 numerical target, not a replacement certificate.

## Acquisition, indexes and execution

`acquire_public_assets.py plan --lock config/official_acquisition_lock.json` lists pinned official assets. Fetch subcommands require explicit official-license acknowledgment and a fresh destination, preserving15GiB free. Large image corpora, pretrained weights and indexes are deliberately excluded. Original executed workers and supporting source are in `execution_source/`; they require official assets and the declared configuration layout. They are not falsely advertised as a portable one-command full hardware rerun. A standalone minimal GPU sample and full dependency map are supplied separately with their actual receipt.

The exact primary model is ColQwen2-v1.0-merged@2c9a09bb37ed19b63eb94ae6c29bb4e76eb6e3c2. ColPali adapter@1b5c8929330df1a66de441a9b5409a878f0de5b0 plus PaliGemma base@30ab955d073de4a91dc5a288e8c97226647e3e5a. No model training is performed. AllGPU execution requires3idle checks and ResourceGuard, not dummy memory reservation. Cache/loading/OCR/index costs are separate from warm query paths.

## Statistical limits and licenses

Pointwise descriptive paired family/source/process intervals; pretraining overlap UNKNOWN. Frozen actual RandomMix is distinct from a retrospective static convex envelope. No new K1, production-IID contract, strict Serving speedup or complete business net benefit is claimed. Epsilon=.05 and delta=.01 remain unchanged. Cal81/Eval73 and old B-L/test/CS stay sealed.

Biomedical official source: vidore/biomedical_lectures_eng_v2@e82eb2f3e9f701688e6f3ba41fffad68c5b3b5fa, CC-BY-NC-SA-4.0. Obtain original slides/queries/qrels from the official source and retain attribution/notices; some copyrighted images were removed in the official release. No original page images or full raw qrels are redistributed. Benchmark-derived output snapshots inherit applicable source conditions; detailed upstream license cards must be retained. ColPali MIT adapter and Gemma base terms are separate. Original software licensing is pending owner approval. Unauthenticated public accessibility and publication are NOT yet completed; they await the owner’s explicit upload approval.
