from pathlib import Path
import csv,json,difflib,re,hashlib,subprocess
B=Path(__file__).parent;O=B/'paper_expanded_001';F=json.loads((O/'numerical_fragments.json').read_text());rows=list(csv.DictReader((B/'results/X2/eab_main_table_dev.csv').open(encoding='utf-8-sig')))
dn={'vidore_v3_hr':'HR','vidore_v3_finance_en':'Finance','vidore_v3_industrial':'Industrial','Pharmaceuticals':'Pharma'}
r100=[];qvals=[]
for d,name in dn.items():
 z=next(r for r in rows if r['dataset']==d and r['action_policy']=='BM25-R100');v=next(r for r in rows if r['dataset']==d and r['action_policy']=='Full-Visual')
 r100.append(f'{name} ($r={float(z["r_pos_mean"]):.4f}$, {float(z["wall_E2E_mean_ms"]):.2f} ms, {100*(1-float(z["wall_E2E_mean_ms"])/float(v["wall_E2E_mean_ms"])):.1f}\\% below FullV)')
 for method in ['Q_regret','QP_regret']:
  z=next(r for r in rows if r['dataset']==d and r['action_policy']==method);qvals.append(f'{name} {"Q" if method=="Q_regret" else "QP"} ${float(z["risk_matched_cost_gain_pp"]):+.2f}$ $[{float(z["paired_family_CI_low_pp"]):.2f},{float(z["paired_family_CI_high_pp"]):.2f}]$')
mapping={'@@MAIN_TABLE@@':F['main_table'],'@@RR_TABLE@@':F['rr_table'],'@@SCATTER@@':F['scatter'],'@@MATCHED@@':F['matched'],'@@R100_VALUES@@':'; '.join(r100),'@@Q_VALUES@@':'; '.join(qvals)}
for src,dst in [('paper_testbed_001.tex','sections/03_method.tex'),('paper_setup_001.tex','sections/04_evaluation.tex'),('paper_results_001.tex','sections/05_results.tex'),('paper_lessons_001.tex','sections/07_limitations_conclusion.tex')]:
 s=(B/src).read_text(encoding='utf8')
 for k,v in mapping.items():s=s.replace(k,v)
 assert '@@' not in s;(O/dst).write_text(s,encoding='utf8')
p=O/'sections/02_problem.tex';s=p.read_text(encoding='utf8').replace('Problem Formulation','Problem Setting and Evaluation Questions')
s+='''
\\paragraph{Evaluation questions.}
RQ1 measures the physical-plan frontier under a fixed visual scorer and examines
candidate omissions. RQ2 separates same-catalog oracle opportunity, realizable
selection, and the cost of decision information. RQ3 asks what calibration labels
and sampling assumptions can establish, separately from descriptive empirical
risk. RQ4 tests whether resident-path costs survive reliable streaming controls.
The three risk statements are distinct: an observed mean on exposed Dev, a
once-only held-out benchmark mean, and a population contract with stated
confidence. The current results satisfy only their explicitly labeled evidence
levels.
''';p.write_text(s,encoding='utf8')
diff=[];hashes={}
for p in sorted(O.rglob('*.tex')):
 old=B/'paper_R3'/p.relative_to(O)
 if not old.exists():continue
 before=old.read_text(encoding='utf8');after=p.read_text(encoding='utf8');hashes[str(p.relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest();diff+=list(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='R3/'+str(p.relative_to(O)),tofile='expanded/'+str(p.relative_to(O))))
(O/'manuscript.diff').write_text(''.join(diff),encoding='utf8');(O/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
words=sum(len(re.findall(r'\b[A-Za-z]+\b',p.read_text(encoding='utf8'))) for p in O.rglob('*.tex'));print(json.dumps({'status':'SOURCE_ASSEMBLED','tex_word_tokens':words,'changed_files':len(hashes),'numerical_rows':len(rows)}))
