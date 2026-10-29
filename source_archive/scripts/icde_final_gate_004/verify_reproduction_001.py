import common as C
import pathlib,json
rows=[]
for tag,stage in [('reproduce_resident_001','S2/resident_analysis_002'),('reproduce_learning_001','S3'),('reproduce_AA_001','S1')]:
 root=C.ART/tag;assert (root/'reproduction_manifest.json').exists(),tag
 for p in (root/'reports'/stage).glob('*.csv'):
  original=C.REPORT/stage/p.name;actual=C.sha(p);expected=C.sha(original);rows.append({'mode':tag,'table':p.name,'original_SHA256':expected,'reproduced_SHA256':actual,'status':'PASS' if actual==expected else 'FAIL','original_path':str(original),'reproduced_path':str(p)})
assert rows;out=C.REPORT/'reproduction_verification_001';out.mkdir(exist_ok=False);C.table(out/'table_hash_checks.csv',rows);C.write(out/'manifest.json',{'time_bj':C.stamp(),'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL','tables':len(rows),'mismatch':sum(x['status']!='PASS' for x in rows),'data':'same immutable raw observations, no new query labels or GPU measurements'});assert all(r['status']=='PASS' for r in rows),[r for r in rows if r['status']!='PASS'];print(json.dumps({'status':'PASS_REPRODUCTION_HASH_IDENTICAL','tables':len(rows),'time_bj':C.stamp()}),flush=True)
