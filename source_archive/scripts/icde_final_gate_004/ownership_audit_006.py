import common as C
import json,pathlib
rows=[];owned={}
for p in C.ART.rglob('resource_guard.json'):
 z=json.loads(p.read_text());owned[z['pid']]=str(p.parent.relative_to(C.ROOT));rows.append({'path':str(p.relative_to(C.ROOT)),'pid':z['pid'],'foreign':z['foreign_pids'],'valid':z['valid']})
for r in rows:r['foreign_known_own_runs']={str(pid):owned.get(pid,'UNKNOWN_EXTERNAL_OR_EXITED') for pid in r['foreign']}
C.write(C.ART/'S1/ownership_audit_006.json',{'time_bj':C.stamp(),'runs':rows});print(json.dumps([r for r in rows if not r['valid']],ensure_ascii=False),flush=True)
