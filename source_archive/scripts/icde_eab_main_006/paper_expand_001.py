"""Generate all numerical LaTeX from versioned CSV; preserve inspected R3 prose."""
from pathlib import Path
import csv,json,math,shutil,re,difflib
B=Path(__file__).parent;SRC=B/'paper_R3';O=B/'paper_expanded_001';O.mkdir(exist_ok=False);(O/'sections').mkdir()
for p in SRC.iterdir():
    if p.is_file():shutil.copyfile(p,O/p.name)
rows=list(csv.DictReader((B/'results/X2/eab_main_table_dev.csv').open(encoding='utf-8-sig')))
domains=['vidore_v3_hr','vidore_v3_finance_en','vidore_v3_industrial','Pharmaceuticals'];names=['HR','Finance','Industrial','Pharma']
short={'Full-Visual':'FullV','BM25-R50':'R50','BM25-R100':'R100','BM25':'BM25','Q_regret':'Q','QP_regret':'QP','RandomMix_Q_regret':'Mix-Q','RandomMix_QP_regret':'Mix-QP'}
f=lambda r,k:float(r[k])
def notes_off(s):
    while '\\OutlineNote{' in s:
        a=s.index('\\OutlineNote{');i=a+len('\\OutlineNote')
        for _ in range(2):
            assert s[i]=='{';n=1;i+=1
            while n:
                if s[i]=='{' and (i==0 or s[i-1]!='\\'):n+=1
                elif s[i]=='}' and (i==0 or s[i-1]!='\\'):n-=1
                i+=1
        s=s[:a]+s[i:]
    return s
for p in (SRC/'sections').glob('*.tex'):(O/'sections'/p.name).write_text(notes_off(p.read_text(encoding='utf8')),encoding='utf8')
main=(SRC/'main.tex').read_text(encoding='utf8').replace('\\showoutlinetrue','\\showoutlinefalse')
start=main.index('% Agreed seven-section');end=main.index('\\input{sections/01_introduction}',start)
main=main[:start]+'% Agreed eight-section EAB structure, Orange 14 R2 / 16 R2.\n'+main[end:]
main=main.replace('% The next two files form ONE Experimental Evaluation section.','% V Experimental Setup; VI Results.').replace('% This file contains Discussion and Limitations, followed by Conclusion.','% VII Lessons; VIII Threats to Validity and Conclusion.')
astart=main.index('This paper studies');aend=main.index('\\end{abstract}',astart)
main=main[:astart]+'''Adaptive retrieval can save work only when its plan catalog offers useful choices,
its decision information identifies those choices, and the selection overhead does
not consume their benefit. We present MDRouter as an experimental testbed for
lexical search, candidate-limited exact visual reranking, and full visual document
retrieval. The current study separates relevance quality, positive regret relative
to a fixed reference, measured execution cost, and statistical evidence. On four
repeatedly used development workloads, larger lexical candidate sets provide
measured cost advantages but do not consistently repair reranking failures;
lightweight learned policies do not establish a stable advantage over strong
static and randomized controls at matched risk. Finite-label calibration and
failed streaming equivalence checks further limit deployable claims. These are
conditional findings: independent evaluation and additional-retriever results
remain explicit evidence gates in this working draft.
'''+main[aend:]
(O/'main.tex').write_text(main,encoding='utf8')
tab=[r'\begin{table*}[t]\centering\scriptsize',r'\caption{Complete resident-path observations on exposed Dev data. Quality uses linear-gain nDCG@10; $r$ is positive reference-relative regret. Gain is the same-risk ideal four-static random envelope minus the measured policy cost, divided by FullV cost. Intervals are pointwise paired family/process bootstrap95\%, not independent-test significance. Only Finance passes the resident A/A gate. Mix-Q and Mix-QP are measured frozen random policies, distinct from the ideal envelope.}\label{tab:main}',r'\begin{tabular}{llrrrrrl}\toprule Domain & Policy & nDCG & $r$ & Harm & E2E(ms) & Gain(pp) & Paired95\% interval\\\midrule']
for d,dn in zip(domains,names):
    for r in rows:
        if r['dataset']!=d or r['action_policy'].startswith('RR-'):continue
        tab.append(f'{dn} & {short[r["action_policy"]]} & {f(r,"nDCG@10"):.4f} & {f(r,"r_pos_mean"):.4f} & {100*f(r,"harm_rate"):.1f}\\% & {f(r,"wall_E2E_mean_ms"):.2f} & {f(r,"risk_matched_cost_gain_pp"):+.2f} & [{f(r,"paired_family_CI_low_pp"):.2f},{f(r,"paired_family_CI_high_pp"):.2f}]'+r'\\')
    tab.append(r'\midrule')
tab+=[r'\bottomrule\end{tabular}\end{table*}']
rrtab=[r'\begin{table}[t]\centering\scriptsize',r'\caption{RR-style adapted controls: frozen Fit81 Q150/Ridge models, hard/soft supervision. Costs are recomposed static paths plus measured feature and CPU inference terms; complete routed-path P95 and CPU remain missing. This is not an original RetrievalRouter reproduction. All mean regrets exceed .05.}\label{tab:rr}',r'\begin{tabular}{llrrr}\toprule Domain & Target & $r$ & Cost(ms) & Gain(pp)\\\midrule']
for r in rows:
    if not r['action_policy'].startswith('RR-'):continue
    rrtab.append(f'{names[domains.index(r["dataset"])]} & {r["action_policy"].split()[-1]} & {f(r,"r_pos_mean"):.4f} & {f(r,"wall_E2E_mean_ms"):.2f} & {f(r,"risk_matched_cost_gain_pp"):+.2f}'+r'\\')
rrtab+=[r'\bottomrule\end{tabular}\end{table}']
def pic(w,h):return [r'\setlength{\unitlength}{1pt}',f'\\begin{{picture}}({w},{h})']
def text(x,y,s,size='scriptsize'):return f'\\put({x:.2f},{y:.2f}){{\\makebox(0,0){{\\{size} {s}}}}}'
def line(x,y,w,h=0):
    if h==0:return f'\\put({x:.2f},{y:.2f}){{\\line(1,0){{{w:.2f}}}}}'
    if w==0:return f'\\put({x:.2f},{y:.2f}){{\\line(0,1){{{h:.2f}}}}}'
    raise ValueError('axis-aligned line only')
# Four-domain risk--cost scatter. Coordinates are generated, never hand edited.
z=pic(476,242)
for di,d in enumerate(domains):
    ox=28+(di%2)*238;oy=25+(1-di//2)*120;W,H=187,82
    stat=[r for r in rows if r['dataset']==d and r['action_policy'] in ['Full-Visual','BM25-R50','BM25-R100','BM25']];mx=max(f(r,'wall_E2E_mean_ms') for r in stat)*1.16;my=max(f(r,'r_pos_mean') for r in stat)*1.13
    z+=[line(ox,oy,W),line(ox,oy,0,H),text(ox+W/2,oy+H+12,names[di]),text(ox+W/2,oy-16,'Warm full-path cost (ms)')]
    for t in [0,.05,.1]:
        if t<=my:z+=[text(ox-14,oy+t/my*H,f'{t:.2f}'),line(ox-2,oy+t/my*H,4)]
    for t in [0,20,40]:
        if t<=mx:z+=[text(ox+t/mx*W,oy-6,str(t)),line(ox+t/mx*W,oy,0,3)]
    for r in stat:
        x=ox+f(r,'wall_E2E_mean_ms')/mx*W;y=oy+f(r,'r_pos_mean')/my*H;z+=[f'\\put({x:.2f},{y:.2f}){{\\circle*{{3.2}}}}',text(x+(10 if r['action_policy']!='BM25-R50' else -10),y+7,short[r['action_policy']],'tiny')]
z+=[r'\end{picture}'];scatter='\n'.join([r'\begin{figure*}[t]\centering',*z,r'\caption{Measured static risk--cost points, each with its own FullV reference. Positive regret and mean nDCG are separate quantities; the table reports both. Exposed Dev, primary ColQwen2 only; secondary-retriever pilot is not implied by this figure.}\label{fig:frontier}\end{figure*}'])
z=pic(470,170);xmin,xmax=-48,31;ox,W=116,330
xx=lambda x:ox+(x-xmin)/(xmax-xmin)*W
for t in [-40,-20,0,20]:z+=[line(xx(t),24,0,130),text(xx(t),14,str(t))]
learn=[r for r in rows if r['action_policy'] in ['Q_regret','QP_regret']]
for i,r in enumerate(learn):
    y=151-i*16;lo,hi=[f(r,k) for k in ['paired_family_CI_low_pp','paired_family_CI_high_pp']];v=f(r,'risk_matched_cost_gain_pp');z+=[text(53,y,names[domains.index(r['dataset'])]+'/'+short[r['action_policy']]),line(xx(lo),y,xx(hi)-xx(lo)),line(xx(lo),y-3,0,6),line(xx(hi),y-3,0,6),f'\\put({xx(v):.2f},{y}){{\\circle{"*" if r["A/A_gate"]=="PASS" else ""}{{4}}}}']
z+=[text(ox+W/2,1,'Cost gain vs same-risk static/random (pp of FullV)'),r'\end{picture}'];matched='\n'.join([r'\begin{figure*}[t]\centering',*z,r'\caption{Learned-policy increments with conditional paired95\% intervals. Open markers denote insufficient resident A/A. Negative values favor the static/random envelope; evidence does not establish a stable learned-policy advantage.}\label{fig:matched}\end{figure*}'])
(O/'numerical_fragments.json').write_text(json.dumps({'main_table':'\n'.join(tab),'rr_table':'\n'.join(rrtab),'scatter':scatter,'matched':matched},indent=2),encoding='utf8')
print('GENERATED_NUMERICAL_FRAGMENTS',len(rows))
