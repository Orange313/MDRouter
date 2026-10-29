"""Post-hoc explanation of fixed policy outputs; no fitting or threshold selection."""
import common as C
import os,sys,collections,json,types
import numpy as np
import pyarrow.parquet as pq
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import engine_exact_004 as E
import learning_base_001 as L
from scipy.stats import spearmanr
C.start('T4_preparation/candidate_prediction_diagnostic_001');proto=C.load(C.ART/'T1/biomedical_frozen_protocol.json');pages=C.load(C.ART/'T1_preparation/ocr_001/pages.json','UNLABELLED_OCR');bm=E.backend()['BM25'](pages);ids=[int(p['corpus_id']) for p in pages]
from aaber.router.fast_query_router import FastLexicalFeatureEncoder
encoder=FastLexicalFeatureEncoder();encoder.fit_corpus([types.SimpleNamespace(searchable_text=x['markdown'] or '') for x in pages]);questions={int(r['query_id']):r['query'] for r in C.load(C.ROOT/proto['query_family_map'])};per=C.load(C.REPORT/'T2/summary_001/biomedical_per_query.csv');actual={(int(x['query_id']),x['method']):x for x in per};ql=collections.defaultdict(dict);path=C.ROOT/proto['official_inputs']['qrels']['path'];assert C.sha(path)==proto['official_inputs']['qrels']['sha256'];C.INPUT[str(path.relative_to(C.ROOT))]={'sha256':C.sha(path),'role':'AUTHORIZED_NEW_PUBLIC_BENCHMARK_POSTHOC_DIAGNOSTIC_NOT_OLD_SEALED_LABELS'}
for r in pq.read_table(path).to_pylist():ql[int(r['query-id'])][int(r['corpus-id'])]=int(r['score'])
locked=C.load(C.ART/'T0/locked_models.json','FROZEN_CHECKPOINTS_NO_FIT');models={}
for r in locked:
    a=[np.array(x) for x in r['model']];a[0]=a[0].astype(bool);models[r['method']]=tuple(a)
cat=np.array([0,2,3]);cost=np.array(proto['inherited_cost_vector']);candidate=[];predictions=[]
for q,text in questions.items():
    order,scores=bm.search(text);ref=json.loads(actual[q,'Full-Visual']['ranking']);positive={p for p in ref if ql[q].get(p,0)>0};qf=encoder.transform(types.SimpleNamespace(question=text));Q=np.array([qf[n] for n in encoder.feature_names]);P=E.probe_features(bm,text,order,scores)
    for plan,k in [('BM25-R50',50),('BM25-R100',100)]:
        missing=positive-{ids[j] for j in order[:k]};r=float(actual[q,plan]['r']);assert r<=1e-10 or missing,'HARM_WITHOUT_POSITIVE_REFERENCE_CANDIDATE_MISS';candidate.append({'query_id':q,'query':text,'plan':plan,'r':r,'signed_gap':actual[q,plan]['signed_gap'],'nDCG10':actual[q,plan]['nDCG10'],'reference_positive_top10':sorted(positive),'omitted_positive_reference_top10':sorted(missing),'candidate_count':min(k,len(order)),'harm_explained_by_positive_reference_omission':r<=1e-10 or bool(missing),'scope':'POST_HOC_DESCRIPTIVE_SHARED_SCORER_DIAGNOSTIC'})
    for method,X in [('Q_regret',Q),('QP_regret',np.r_[Q,P])]:
        pred=L.predict(models[method],X[None])[0];a=int(cat[np.argmin(cost[cat]+20*np.maximum(pred[cat],0))]);chosen=C.PLANS[a];row=actual[q,method];assert chosen==row['action'],'FROZEN_ACTION_COMPATIBILITY_MISMATCH';r=float(row['r']);predictions.append({'query_id':q,'query':text,'method':method,'action':chosen,'predicted_r_raw':float(pred[a]),'predicted_r_nonnegative':float(max(pred[a],0)),'actual_r':r,'prediction_error':r-max(float(pred[a]),0),'False_Safe_diagnostic':chosen!='Full-Visual' and r>.05,'reference_nDCG10':row['reference_nDCG10'],'nDCG10':row['nDCG10'],'BM25_Recall50':row['BM25_Recall50'],'scope':'POST_HOC_FIXED_MODEL_DIAGNOSTIC; event r>.05 is not a per-query certificate violation'})
C.table(C.R/'candidate_miss_explanation.csv',candidate);C.table(C.R/'frozen_prediction_errors.csv',predictions)
summary=[]
for m in ['Q_regret','QP_regret']:
    rr=[r for r in predictions if r['method']==m];safe=[r for r in rr if r['False_Safe_diagnostic']];summary.append({'method':m,'questions':len(rr),'False_Safe_queries':len(safe),'False_Safe_regret_mass_fraction':sum(x['actual_r'] for x in safe)/sum(x['actual_r'] for x in rr),'predicted_actual_spearman':float(spearmanr([x['predicted_r_nonnegative'] for x in rr],[x['actual_r'] for x in rr]).statistic),'scope':'POST_HOC_DESCRIPTIVE_NO_FIT'})
C.table(C.R/'prediction_summary.csv',summary)
cases=sorted([r for r in predictions if r['method']=='QP_regret' and r['False_Safe_diagnostic']],key=lambda r:-r['actual_r'])[:10]
C.prose(C.R/'false_safe_cases.md','# 固定模型的事后错误案例\n\n'+ '\n'.join(f'- q{x["query_id"]}: {x["query"]}\n  action={x["action"]}, predicted r={x["predicted_r_nonnegative"]:.4f}, actual r={x["actual_r"]:.4f}, BM25 Recall50={x["BM25_Recall50"]}。' for x in cases)+'\n\n全部逐查询预测误差/候选遗漏归档，未删除任何高伤害查询；不以这些结果挑模型或阈值。支持F1候选机制与F2信息误差解释，不作因果或正式认证声明。\n');print(C.clean({'prediction':summary,'candidate_harmed_queries':sum(x['r']>1e-10 for x in candidate),'all_harm_has_positive_reference_omission':all(x['harm_explained_by_positive_reference_omission'] for x in candidate)}))
