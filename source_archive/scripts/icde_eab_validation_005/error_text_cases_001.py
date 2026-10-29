import common as C,json
C.start('A1/text_cases_001');rows=C.load(C.REPORT/'A1/false_safe_cases.csv');queries={}
for d in C.DOM[:3]:
 for q in C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json'):queries[d,int(q['query_id'])]=q['query']
for q in C.load(C.PREV/'E2/allowed_fitdev_queries.json'):
 if q['role']=='development':queries['Pharmaceuticals',int(q['query_id'])]=q['query']
out=[];lines=['# EXPLORATORY｜完整False-Safe查询及机制线索','所有已登记False-Safe均列出，不仅挑选高损失例；查询来自授权Dev。r>.05为逐查询诊断，不等于平均风险合同违约。']
for r in rows:
 d=r['domain'];q=int(r['query_id']);text=queries[d,q];act=json.loads(r['actions']);miss=float(r['mean_omitted_positive_reference_pages']) if r['mean_omitted_positive_reference_pages'] else None
 mechanism='Observed candidate omission accompanies loss; association does not establish unique cause' if miss is not None and miss>0 else 'No omitted reference-positive candidate observed; lexical ranking or policy choice remains a possible mechanism, not identified causality'
 out.append({**r,'query_text':text,'mechanism_evidence':mechanism});lines += [f'\n## {d} q{q} / {r["method"]}',f'> {text}',f'actions={act}; reference nDCG={r["reference_nDCG10"]}; chosen nDCG={r["nDCG10"]}; positive regret={r["positive_regret"]}; full-path E2E={r["actual_E2E_s"]}s; omitted reference-positive mean={miss}.',mechanism]
C.table(C.R/'all_false_safe_query_text.csv',out);C.prose(C.R/'error_case_analysis_with_text.md','\n\n'.join(lines)+'\n');C.write(C.A/'complete.json',{'status':'PASS_AUTHORIZED_DEV_CASES','n':len(out),'time_bj':C.stamp(),'sealed_reads':0});print(json.dumps({'status':'PASS_TEXT_CASES','n':len(out)}),flush=True)
