from pathlib import Path
import json,csv
B=Path(__file__).parent;out=B/'paper_update_007_003';out.mkdir(exist_ok=False)
prior=json.loads((B/'paper_update_007_002/hunks.json').read_text(encoding='utf8'))
files={x['logicalPath']:x['newText'] for x in prior}
snap=json.loads((B/'authority/paper_R4_snapshot.json').read_text(encoding='utf8'))
method=snap['files']['sections/03_method.tex']['content']
old='''The implementation stores page embeddings in half precision and performs exact
scoring in single precision. Padded passage vectors are zero, and the scorer
applies the corresponding nonnegative maximum. Rerank and FullV share this
implementation, token handling, and numerical conventions. Rankings use descending
score followed by ascending numerical page ID, so ties have a stable outcome.'''
new='''Page embeddings are stored in half precision and scoring uses single precision.
Historical primary-model runs used zero-padded maxima with a floor at zero;
those outputs remain explicitly labeled legacy results. The new Biomedical and
ColPali executor masks true padding to minus infinity and retains negative
valid-token maxima, implementing Equation~\\ref{eq:maxsim}. Within each run,
Rerank and FullV share the scorer, token handling, and numerical conventions.
Rankings use descending score then ascending numerical page ID. A supplementary
four-Dev audit checks utility/risk parity across executor versions; it does not
transfer latency results or change the historical 005 protocol.'''
assert old in method;files['sections/03_method.tex']=method.replace(old,new)
files['sections/04_evaluation.tex']=files['sections/04_evaluation.tex'].replace('an separately timestamped','a separately timestamped')
rows=list(csv.DictReader((B/'results_007/T3/cross_retriever_main_table.csv').open(encoding='utf8')))
short={'Full-Visual':'FullV','BM25-R50':'R50','BM25-R100':'R100','BM25':'BM25'}
names={'vidore_v3_hr':'HR','vidore_v3_finance_en':'Finance','vidore_v3_industrial':'Industrial','Pharmaceuticals':'Pharma'}
z=[r'\begin{table*}[t]\centering\scriptsize',r'\caption{New ColPali-v1.3 static measurements on the same exposed four-domain Dev questions, using each model\textquotesingle s own FullV reference. Brackets are pointwise 95\% family/process bootstrap intervals. Three processes and three timing repeats do not add independent queries. Actual complete warm paths and exact valid-token MaxSim; no new certification.}\label{tab:colpali}',r'\begin{tabular}{llrrrrrr}\toprule Domain & Plan & $n$ & nDCG & $r$ [95\% CI] & E2E(ms) [95\% CI] & GPU(ms) & P95(ms)\\\midrule']
for r in rows:
    f=lambda k:float(r[k])
    z.append(f'{names[r["dataset"]]} & {short[r["plan"]]} & {r["queries"]} & {f("nDCG10"):.4f} & {f("mean_r"):.4f} [{f("r_CI_low"):.4f},{f("r_CI_high"):.4f}] & {f("warm_E2E_ms"):.2f} [{f("E2E95_low_ms"):.2f},{f("E2E95_high_ms"):.2f}] & {1000*f("complete_path_GPU_s"):.3f} & {f("P95_ms"):.2f}'+r'\\')
z.append(r'\bottomrule\end{tabular}\end{table*}')
# Avoid requiring an extra text-symbol package.
table='\n'.join(z).replace(r'model\textquotesingle s',"model's")
p='sections/05_results.tex';t=files[p];a=t.index(r'\begin{table*}',t.index(r'\subsection{Additional Retriever'));b=t.index(r'\end{table*}',a)+len(r'\end{table*}');files[p]=t[:a]+table+t[b:]
for p in ['sections/03_method.tex','sections/04_evaluation.tex','sections/05_results.tex']:
    q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(files[p],encoding='utf8')
(out/'postimages.json').write_text(json.dumps({p:files[p] for p in ['sections/03_method.tex','sections/04_evaluation.tex','sections/05_results.tex']},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'files':3,'ColPali_rows':len(rows),'purpose':'correct scorer-version description and add paired uncertainty; no numeric or method selection'}))
