# Portability and evidence boundaries

The CPU replay verifies 15 unchanged007 numerical CSVs and 15 new008 paired/ColPali CSVs (30 total), and redraws three figure families. The portable008 scripts differ only in removing original full-project inventory/code-location assertions; all shipped inputs are independently hash-checked before replay. Exact original executed sources are separately archived. No statistical formula, bootstrap seed, query group, method, or numerical row is changed. Empty-OCR sensitivity is archived but requires official qrels to recompute and is not counted among these30.

The minimal GPU command is:

    python minimal_gpu_sample.py --gpu 0 --adapter ADAPTER_SNAPSHOT --base BASE_SNAPSHOT --corpus CORPUS_PARQUET --queries QUERIES_PARQUET --ocr OCR_PAGES_JSON --output outputs/gpu_smoke_001

It requires the pinned GPU environment, official external assets, three idle checks and ResourceGuard. The completed smoke used source from clean checkout5012b5cbfaf791fe6a1bed44809287a20f955a72, the existing pinned GPU dependencies,16 pages/one query/12 timings, and zero labels. It is not a fresh GPU dependency installation or a quality benchmark. The separate CPU dependency environment was freshly installed and the original checkout was not imported. Resource guards and all repeated request rows are retained, including the failed pre-CUDA process2 and its successful new-directory retry.

Original-host paths in public snapshots are replaced with PROJECT_ROOT/USER_HOME placeholders. The original SHA and public SHA are both recorded; numerical results are unchanged. Paths are not credentials. CPU and GPU overlap are not added, and warm E2E is not complete business net benefit or validated Serving throughput.

OCR recipe: `python prepare_ocr.py --corpus CORPUS_PARQUET --output outputs/ocr_001`; compare the recorded Tesseract runtime with config/ocr_runtime_lock.json. This portable CLI is supplied for acquisition convenience and is syntax-checked, not advertised as a newly measured exact OCR rerun. The original executed OCR recipe is archived separately.
