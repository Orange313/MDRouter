"""Rebuild review011 mechanism counts and declared illustrative cases from public replay CSV."""
from pathlib import Path
import argparse,csv,hashlib,json
def generate(rows):
    ids=sorted({int(x['query_id']) for x in rows});assert len(ids)==160
    q={i:{x['method']:x for x in rows if int(x['query_id'])==i} for i in ids};f=lambda x,k:float(x[k]);coverage=[]
    for plan in ['BM25-R50','BM25-R100']:
        rr=[q[i][plan] for i in ids];covered=[x for x in rr if x['reference_top10_in_candidates']=='True'];harm=[x for x in rr if f(x,'r')>1e-10];assert all(json.loads(x['ranking'])==json.loads(x['reference_ranking']) and f(x,'r')<=1e-10 for x in covered)
        coverage.append({'plan':plan,'queries':160,'reference_top10_covered':len(covered),'covered_rankings_equal':len(covered),'harm_queries':len(harm),'incomplete_reference_coverage':160-len(covered),'incomplete_but_unharmed':160-len(covered)-len(harm),'scope':'frozen descriptive diagnostic, not an online feature or certification'})
    empty_ids={19,21,22,25,87,154} # Frozen009emptyOCR sensitivity membership, not newly read qrels.
    rules=[('R50_REPAIRED_BY_R100',lambda i:f(q[i]['BM25-R50'],'r')>.05 and f(q[i]['BM25-R100'],'r')<1e-10,lambda i:f(q[i]['BM25-R50'],'r')-f(q[i]['BM25-R100'],'r')),
      ('QP_FALLBACK_AVOIDS_R100_HARM',lambda i:f(q[i]['BM25-R100'],'r')>.1 and q[i]['QP_regret']['action']=='Full-Visual',lambda i:f(q[i]['BM25-R100'],'r')),
      ('QP_FALSE_SAFE',lambda i:f(q[i]['QP_regret'],'r')>.05 and q[i]['QP_regret']['action']!='Full-Visual',lambda i:f(q[i]['QP_regret'],'r')),
      ('EMPTY_OCR_ASSOCIATED',lambda i:i in empty_ids,lambda i:f(q[i]['BM25-R100'],'r')),
      ('PROBE_RISK_GAIN_HAS_COST',lambda i:f(q[i]['Q_regret'],'r')-f(q[i]['QP_regret'],'r')>=.05 and f(q[i]['QP_regret'],'E2E_mean_s')>f(q[i]['Q_regret'],'E2E_mean_s'),lambda i:f(q[i]['QP_regret'],'E2E_mean_s')-f(q[i]['Q_regret'],'E2E_mean_s'))]
    remaining=set(ids);cases=[]
    for name,eligible,key in rules:
        eligible_ids=sorted([i for i in remaining if eligible(i)],key=lambda i:(-key(i),i));assert eligible_ids;i=eligible_ids[0];remaining.remove(i);r=q[i]
        cases.append({'case':name,'query_id':i,'source_group':r['Full-Visual']['source_group'],'family':r['Full-Visual']['family'],'R50_r':f(r['BM25-R50'],'r'),'R100_r':f(r['BM25-R100'],'r'),'QP_r':f(r['QP_regret'],'r'),'Q_r':f(r['Q_regret'],'r'),'QP_action':r['QP_regret']['action'],'R100_reference_containment':r['BM25-R100']['reference_top10_in_candidates'],'emptyOCR_affected':i in empty_ids,'eligible_distinct_cases':len(eligible_ids),'scope':'posthoc max-diagnostic-score illustration, not causal/prevalence evidence; all160retained'})
    return coverage,cases
def main():
    p=argparse.ArgumentParser();p.add_argument('--per-query-csv',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.per_query_csv);rows=list(csv.DictReader(source.open(encoding='utf-8-sig')));out=Path(a.output);out.mkdir(parents=True,exist_ok=False);coverage,cases=generate(rows)
    for name,data in [('candidate_coverage.csv',coverage),('mechanism_cases.csv',cases)]:
        with (out/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'mechanism_receipt.json').write_text(json.dumps({'input_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'queries':160,'case_ids':[x['query_id'] for x in cases],'GPU':0,'new_qrels_reads':0,'new_models_or_thresholds':False},indent=2)+'\n')
if __name__=='__main__':main()
