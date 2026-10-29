import common as C
import subprocess,json
out=C.ART/'Y1/BM25_CPU_design_001';out.mkdir(parents=True,exist_ok=False)
C.write(out/'design.json',{'time_bj':C.stamp(),'role':'supplemental cost-only engineering detail, not risk/model/threshold change','reason':'measure BM25-only cold path without charging unused visual model/index; FullV standalone scenario excludes unused OCR','queries':'same frozen160 IDs; no qrels','processes':3,'warm_requests':480,'seed':C.SEED,'new_quality_or_certification':False,'code_SHA256':C.sha(C.ROOT/'scripts'/C.RUN/'y1_bm25_cpu_cost_001.py')})
for p in range(3):
    with (out/f'p{p}.log').open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),str(C.ROOT/'scripts'/C.RUN/'y1_bm25_cpu_cost_001.py'),str(p)],stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0
print('PASS_ALL_THREE_CPU_ONLY_PROCESSES',flush=True)
