import common as C
import numpy as np,types,ast,math,collections,sys,json
sys.path.insert(0,str(C.ROOT/'src'))
from aaber.router.fast_query_router import FastLexicalFeatureEncoder
from aaber.common.text import tokenize
from mdrouter_v5.graded_metrics import stable_ranking
from engine_001 import probe_features
C.start('E4/feature_semantics_audit_002');C.STAGE='E4_FEATURE_SEMANTICS'
source=C.load('scripts/cross_domain_four_plan_001.py');node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='BM25');env={'CFG':C.load('configs/hr_pilot_sparse_004.json'),'np':np,'Counter':collections.Counter,'defaultdict':collections.defaultdict,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module([node],type_ignores=[]),'frozen_BM25','exec'),env);rows=[]
for d,ix in zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003']):
 pages=C.load('cache/indexes/'+ix+'/pages.json');qs=C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json');cached=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz');enc=FastLexicalFeatureEncoder();enc.fit_corpus([types.SimpleNamespace(searchable_text=p['markdown'] or '') for p in pages]);bm=env['BM25'](pages);lut={int(q):i for i,q in enumerate(cached['qids'])};qq=[];pp=[];want=[]
 for q in qs:
  f=enc.transform(types.SimpleNamespace(question=q['query']));qq.append([f[k] for k in enc.feature_names]);p=bm.search(q['query']);pp.append(probe_features(bm,q['query'],*p));want.append(lut[int(q['query_id'])])
 Q=np.array(qq);P=np.array(pp);qpass=np.allclose(Q,cached['Q'][want],rtol=1e-5,atol=1e-6);ppass=np.allclose(P,cached['P'][want],rtol=1e-5,atol=1e-6);rows.append({'domain':d,'Q_max_absolute_difference':float(np.max(np.abs(Q-cached['Q'][want]))),'P_max_absolute_difference':float(np.max(np.abs(P-cached['P'][want]))),'Q_PASS':bool(qpass),'P_PASS':bool(ppass)})
assert all(r['Q_PASS'] and r['P_PASS'] for r in rows);C.write(C.A/'results.json',rows);C.finish('PASS_CACHED_ONLINE_FEATURE_SEMANTICS',json.dumps(rows),'仅浮点舍入差，P完全一致；不重训、不改工作点。')
