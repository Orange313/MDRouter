"""End-to-end role filter / policy execution / analysis mock; no real release path."""
from pathlib import Path
import json, numpy as np, collections, sys
def run(out,mode='mock',provider=None,backend=None):
    # There is no authorization to real Eval in this run. No data-provider call before this gate.
    if mode!='mock':return {'status':'DENIED_BEFORE_PROVIDER_CALL','exit_code':77,'real_payload_reads':0}
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(202610272130)
    fake=[{'id':i,'role':role,'family':'f'+str(i//2),'text':'SYNTHETIC SCHEMA QUERY '+str(i)} for role,start,n in [('fit',0,81),('development',1000,45),('calibration',2000,81),('evaluation',3000,73)] for i in range(start,start+n)]
    ids={r['id'] for r in fake if r['role']=='evaluation'}
    calls=[]
    def acquire(kind,requested):
        assert set(requested)==ids and len(requested)==73
        calls.append({'kind':kind,'requested_ids':sorted(requested),'scope':'MOCK_EVAL_ONLY'})
        return [r for r in fake if r['id'] in requested]
    q=acquire('queries',ids);j=acquire('qrels',ids);assert len(q)==len(j)==73
    plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];methods=plans+['RandomMix_Q_regret','RandomMix_QP_regret','Q_regret','QP_regret','RR-style adapted hard','RR-style adapted soft']
    rows=[]
    for r in q:
        loss=rng.uniform(0,.2,4);loss[0]=0
        for m in methods:
            act=plans.index(m) if m in plans else int(rng.integers(4))
            rows.append({'query_id':r['id'],'family':r['family'],'method':m,'action':plans[act],'r':float(loss[act]),'nDCG10':float(1-loss[act]),'E2E_s':float(.04 if act==0 else .02),'scope':'SYNTHETIC_MOCK_NOT_SCIENTIFIC_EVIDENCE'})
    with (out/'mock_per_query.json').open('x') as f:json.dump(rows,f)
    contrasts=[];families=np.array([r['family'] for r in q]);groups=np.unique(families)
    boots=[np.concatenate([np.flatnonzero(families==g) for g in rng.choice(groups,len(groups),replace=True)]) for _ in range(2000)]
    for method,ref in [('Q_regret','BM25-R100'),('QP_regret','Q_regret')]:
        for metric in ['r','E2E_s']:
            diff=np.array([next(x[metric] for x in rows if x['query_id']==r['id'] and x['method']==method)-next(x[metric] for x in rows if x['query_id']==r['id'] and x['method']==ref) for r in q]);lo,hi=np.quantile([diff[ix].mean() for ix in boots],[.025,.975]);contrasts.append(dict(method=method,reference=ref,metric=metric,mean=float(diff.mean()),low=float(lo),high=float(hi)))
    (out/'mock_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    receipt={'status':'PASS_MOCK_ROLE_FILTER_EXECUTION_ANALYSIS','mock_queries':73,'methods':10,'rows':len(rows),'real_Eval_reads':0,'real_Cal_reads':0,'provider_calls':calls,'hardware_integration':'NOT_RUN','future_execution':'requires separately bound real role-filtered provider, pinned hardware adapter and explicit user authorization; no authorization flag accepted in current run'}
    (out/'mock_receipt.json').write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    r=run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'mock');print(json.dumps({k:v for k,v in r.items() if k!='provider_calls'}));sys.exit(r.get('exit_code',0))
