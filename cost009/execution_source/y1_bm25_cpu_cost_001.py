import time
EARLY=time.perf_counter()
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import common as C
import ast,collections,math,sys
import numpy as np
import pyarrow.parquet as pq
p=int(sys.argv[1]);C.start(f'Y1/BM25_CPU_p{p}_001');sys.path.insert(0,str(C.ROOT/'src'))
from aaber.common.text import tokenize
from mdrouter_v5.graded_metrics import stable_ranking
pages=C.load('artifacts/icde_eab_independent_007/T1_preparation/ocr_001/pages.json');pages.sort(key=lambda x:int(x['corpus_id']));ids=[int(x['corpus_id']) for x in pages];proto=C.load('artifacts/icde_eab_independent_007/T1/biomedical_frozen_protocol.json');qp=C.ROOT/proto['official_inputs']['queries']['path'];assert C.sha(qp)==proto['official_inputs']['queries']['sha256'];queries={int(x['query-id']):x['query'] for x in pq.read_table(qp).to_pylist()}
nodes=[x for x in ast.parse(C.load('scripts/cross_domain_four_plan_001.py')).body if isinstance(x,ast.ClassDef) and x.name=='BM25'];env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'frozen_BM25','exec'),env)
ct=time.process_time();st=time.perf_counter();bm=env['BM25'](pages);buildwall=time.perf_counter()-st;buildCPU=time.process_time()-ct
st=time.perf_counter();ct=time.process_time();bm.search(queries[proto['query_ids'][0]]);firstwall=time.perf_counter()-st;firstCPU=time.process_time()-ct;cold=time.perf_counter()-EARLY
for _ in range(3):bm.search(queries[proto['query_ids'][0]])
work=proto['query_ids'].copy();np.random.default_rng(C.SEED+p).shuffle(work);rows=[]
for q in work:
    st=time.perf_counter();ct=time.process_time();out=bm.search(queries[q]);rows.append({'model':'shared_text','process':p,'query_id':q,'plan':'BM25','E2E_s':time.perf_counter()-st,'CPU_s':time.process_time()-ct,'GPU_s':0,'GPU_scope':'STRUCTURAL_CPU_ONLY_NO_CUDA','cohort':'009_NEW_CPU_ONLY_COST','qrels_reads':0})
C.table(C.R/'requests.csv',rows);C.write(C.A/'completed.json',{'status':'PASS_CPU_ONLY_COST','process':p,'queries':160,'BM25_build_wall_s':buildwall,'BM25_build_CPU_s':buildCPU,'first_request_wall_s':firstwall,'first_request_CPU_s':firstCPU,'process_cold_to_first_s':cold,'process_startup_excluding_first_s':cold-firstwall,'GPU_s':0,'GPU_scope':'NO_CUDA_IMPORTED_OR_EXECUTED','sealed_reads':0,'qrels_reads':0,'time_bj':C.stamp()});print({'process':p,'status':'PASS_CPU_ONLY_BM25','requests':len(rows)},flush=True)
