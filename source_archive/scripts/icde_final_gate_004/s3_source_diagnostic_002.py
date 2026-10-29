import common as C
import numpy as np,collections,json
C.A=C.ART/'S3';C.R=C.REPORT/'S3';C.STAGE='S3';a=C.load(C.PREV/'E2/pharma_dataset.npz');qs=C.load(C.PREV/'E2/allowed_fitdev_queries.json');qd={int(q['query_id']):q for q in qs};rows=[]
for role in ['fit','development']:
 for generator in ['ALL','human','sdg']:
  v=np.array([r==role and (generator=='ALL' or qd[int(q)].get('generator')==generator) for q,r in zip(a['qids'],a['roles'])]);y=a['Y'][v];rows.append({'role':role,'official_generator':generator,'n':int(v.sum()),'candidate_families':len(set(a['family'][v])),'BM25_mean_r':y[:,3].mean(),'BM25_harm_fraction':np.mean(y[:,3]>1e-10),'BM25_r_gt05_fraction':np.mean(y[:,3]>.05),'R100_mean_r':y[:,2].mean(),'R100_harm_fraction':np.mean(y[:,2]>1e-10),'R100_r_gt05_fraction':np.mean(y[:,2]>.05),'scope':'official human/sdg provenance is not IID proof, SDG not independently human-adjudicated by this project'})
C.table(C.R/'Fit_Dev_source_difficulty.csv',rows)
pred=C.load(C.PR/'E2/per_query_policy.csv');pr=collections.defaultdict(list)
for r in pred:
 if r['source']=='Pharma_only' and float(r['kappa'])==10 and r['method'] in ['Q_regret','QP_regret'] and int(r['Pharma_fit_n']) in [20,81]:pr[r['method'],int(r['query_id']),int(r['Pharma_fit_n'])].append(r)
examples=[]
for method in ['Q_regret','QP_regret']:
 ids=sorted({q for m,q,n in pr if m==method})
 for q in ids:
  lo=pr[method,q,20];hi=pr[method,q,81];r0=np.mean([float(r['r']) for r in lo]);r1=np.mean([float(r['r']) for r in hi]);examples.append({'method':method,'query_id':q,'question':qd[q]['query'],'official_generator':qd[q].get('generator','UNKNOWN'),'family':qd[q]['family_id'],'mean_r_n20':r0,'mean_r_n81':r1,'mean_r_change':r1-r0,'actions20':[r['action'] for r in lo],'actions81':[r['action'] for r in hi],'pred20':[json.loads(r['prediction']) for r in lo],'pred81':[json.loads(r['prediction']) for r in hi],'status':'DESCRIPTIVE_FIXED_KAPPA10_NOT_NEW_MODEL_SELECTION'})
C.table(C.R/'per_query_training_change_examples.csv',examples)
text='# S3 source/difficulty sensitivity\n\nOfficial Fit/Dev includes both human and sdg query generators. This is benchmark provenance, not proof that SDG was independently human reviewed. No unreviewed internal synthetic labels were added. All official source strata and allqueries retained; no source-based filtering/reallocation.\n\n|Role|Generator|Queries|BM25 mean r|R100 mean r|\n|---|---|---:|---:|---:|\n'
for r in rows:text+=f'|{r["role"]}|{r["official_generator"]}|{r["n"]}|{r["BM25_mean_r"]:.4f}|{r["R100_mean_r"]:.4f}|\n'
text+='\nLargest positive and negative training changes, preset kappa10 illustration (all cases in CSV):\n\n'
for method in ['Q_regret','QP_regret']:
 z=sorted([r for r in examples if r['method']==method],key=lambda r:r['mean_r_change']);chosen=z[:3]+z[-3:]
 for r in chosen:text+=f'- {method}, Dev query{r["query_id"]} ({r["official_generator"]}): r20={r["mean_r_n20"]:.4f}, r81={r["mean_r_n81"]:.4f}; actions20={r["actions20"]}, actions81={r["actions81"]}.\n'
text+='\nNo source stratum is removed. This decomposes examples, not causality. Formal sampling/IID remains unresolved; Cal/Eval untouched.\n'
C.prose(C.R/'source_difficulty_sensitivity.md',text);print(json.dumps({'status':'PASS_DEV_SOURCE_DIAGNOSTIC','strata':rows,'time_bj':C.stamp()}),flush=True)
