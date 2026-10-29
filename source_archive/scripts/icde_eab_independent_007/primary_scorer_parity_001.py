"""Post-discovery correctness supplement on exposed Dev only; never replaces frozen005."""
import common as C
import os,sys,collections,json
import numpy as np
gpu=int(sys.argv[1]);domain=sys.argv[2];C.start(f'correctness_supplement/{domain}_001');os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import engine_exact_004 as E
E.idle(gpu);sys.path.insert(0,str(C.ROOT/'src'))
from mdrouter_v5.gpu_accounting import ResourceGuard
from mdrouter_v5.graded_metrics import ndcg
guard=ResourceGuard(gpu,C.A/'resource_guard.json',period=.1)
try:
    import torch
    index=dict(zip(C.DOM,['hr_primary_001','finance_primary_001','industrial_primary_003','pharma_fixedcorpus_index_001']))[domain];eng=E.Engine(index,guard)
    if domain=='Pharmaceuticals':
        queries=[x for x in C.load('artifacts/icde_evidence_closure_003/E2/allowed_fitdev_queries.json') if x['role'].lower()=='dev'];labels=C.load('artifacts/icde_evidence_closure_003/E2/public_projection_final_001/fitdev_qrels.json')
    else:
        queries=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_queries.json');labels=C.load(f'artifacts/splits/query_generalization_002/{domain}/dev_qrels.json')
    ql=collections.defaultdict(dict)
    for r in labels:ql[int(r['query_id'])][int(r['corpus_id'])]=int(r['score'])
    old=C.load('reports/icde_final_gate_004/S2/resident_analysis_002/all_unprofiled_requests.csv','EXPOSED_DEV_HISTORICAL_MEASUREMENTS');old={(int(r['query_id']),r['method']):(float(r['nDCG10']),float(r['r'])) for r in old if r['domain']==domain and r['method'] in C.PLANS};rows=[]
    with torch.inference_mode():
        for q in queries:
            guard.check();qid=int(q['query_id']);ref=eng.execute('Full-Visual',q['query']);u0=ndcg(ref,ql[qid])
            for p in C.PLANS:
                rank=ref if p=='Full-Visual' else eng.execute(p,q['query']);u=ndcg(rank,ql[qid]);r=max(u0-u,0);ou,orr=old[qid,p];rows.append({'domain':domain,'query_id':qid,'plan':p,'exact_unclipped_nDCG10':u,'exact_unclipped_r':r,'historical_frozen_nDCG10':ou,'historical_frozen_r':orr,'U_difference':u-ou,'r_difference':r-orr,'utility_risk_parity':abs(u-ou)<1e-10 and abs(r-orr)<1e-10,'ranking':rank,'reference_ranking':ref,'scope':'EXPOSED_DEV_CORRECTNESS_ONLY; no new E2E claim or certificate'})
    C.table(C.R/'primary_scorer_parity.csv',rows);C.write(C.A/'result.json',{'domain':domain,'queries':len(queries),'plan_query_pairs':len(rows),'identical_utility_and_risk':sum(x['utility_risk_parity'] for x in rows),'changed_pairs':sum(not x['utility_risk_parity'] for x in rows),'max_abs_U_diff':max(abs(x['U_difference']) for x in rows),'max_abs_r_diff':max(abs(x['r_difference']) for x in rows),'scope':'legacy implementation retained; new exact valid-token scorer compared only on allowed exposed Dev; does not reopen BL or replace005','time_bj':C.stamp()})
finally:guard.close()
print(json.dumps({'domain':domain,'queries':len(queries),'changed_pairs':sum(not x['utility_risk_parity'] for x in rows),'time_bj':C.stamp()}),flush=True)
