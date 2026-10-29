"""Frozen descriptive evaluator. It refuses real input before an explicit future scope receipt."""
import argparse,pathlib,json,hashlib,csv,sys,collections,numpy as np
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol');p.add_argument('--authorization');p.add_argument('--input');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if not a.smoke:
  if not a.authorization:print('DENIED: explicit future user Eval73-only authorization missing; input not opened');return 77
  auth=json.loads(pathlib.Path(a.authorization).read_text(encoding='utf8'))
  if not (auth.get('explicit_user_approval') is True and auth.get('scope')=='Pharma_Eval73_ONLY' and auth.get('protocol_SHA256')==sha(a.protocol) and auth.get('user_instruction_quote') and auth.get('source_record_id')):print('DENIED: receipt scope/binding incomplete; no input opened');return 77
  proto=json.loads(pathlib.Path(a.protocol).read_text(encoding='utf8'));rows=list(csv.DictReader(pathlib.Path(a.input).open(encoding='utf8')));assert len(set(r['query_id'] for r in rows))==73
 else:
  rows=[dict(query_id=q,family='s'+str(q),method=m,r=0 if m=='Full-Visual' else .01*q,nDCG10=1 if m=='Full-Visual' else 1-.01*q,E2E_s=.04 if m=='Full-Visual' else .03,CPU_s=.02,GPU_activity_s=.01,process=0,repeat=0) for q in range(4) for m in ['Full-Visual','BM25-R100','Q_regret']];proto={'primary_contrasts':[]}
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);methods=sorted(set(r['method'] for r in rows));qids=sorted(set(str(r['query_id']) for r in rows));per=[]
 for q in qids:
  for m in methods:
   z=[r for r in rows if str(r['query_id'])==q and r['method']==m];assert z
   per.append(dict(query_id=q,family=z[0]['family'],method=m,**{k:np.mean([float(x[k]) for x in z]) for k in ['r','nDCG10','E2E_s','CPU_s','GPU_activity_s']}))
 def table(name,z):
  with (out/name).open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
 table('per_query_frozen_policy.csv',per);table('descriptive_main_table.csv',[dict(method=m,**{k:np.mean([r[k] for r in per if r['method']==m]) for k in ['r','nDCG10','E2E_s','CPU_s','GPU_activity_s']},scope='SYNTHETIC_SMOKE' if a.smoke else 'ONE_SHOT_DESCRIPTIVE_NOT_CERTIFIED') for m in methods]);groups=np.array([next(r['family'] for r in per if r['query_id']==q) for q in qids]);rng=np.random.default_rng(202610271730);unique=np.unique(groups);ixs=[np.concatenate([np.flatnonzero(groups==h) for h in rng.choice(unique,len(unique),replace=True)]) for _ in range(10000)];contrasts=[]
 pairs=[('Q_regret','BM25-R100'),('QP_regret','Q_regret'),('Q_regret','RandomMix_Q_regret'),('QP_regret','RandomMix_QP_regret')]+[(m,'Full-Visual') for m in methods if m!='Full-Visual']
 for x,y in pairs:
  if x not in methods or y not in methods:continue
  for k in ['r','nDCG10','E2E_s']:
   d=np.array([next(r[k] for r in per if r['query_id']==q and r['method']==x)-next(r[k] for r in per if r['query_id']==q and r['method']==y) for q in qids]);lo,hi=np.quantile([d[ix].mean() for ix in ixs],[.025,.975]);contrasts.append(dict(method=x,reference=y,metric=k,mean_difference=d.mean(),paired_family95_low=lo,paired_family95_high=hi,scope='conditional fixed benchmark, pointwise, no PAC'))
 table('prespecified_paired_contrasts.csv',contrasts);false=[dict(query_id=r['query_id'],family=r['family'],method=r['method'],r=r['r'],FalseSafe=r['r']>.05) for r in per if r['method'] in ['Q_regret','QP_regret']];table('false_safe.csv',false)
 (out/'access_audit.json').write_text(json.dumps(dict(mode='SYNTHETIC_SMOKE' if a.smoke else 'EXPLICIT_USER_AUTHORIZATION_BOUND',input_SHA256=None if a.smoke else sha(a.input),Cal_reads=0,oldBL_test_CS_reads=0,new_certification='NOT_CERTIFIED'),indent=2),encoding='utf8');print('PASS_ANALYSIS_SCHEMA_ONLY' if a.smoke else 'DESCRIPTIVE_EVAL_COMPLETE_NOT_CERTIFIED');return 0
if __name__=='__main__':sys.exit(main())
