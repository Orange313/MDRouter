import common as C
import collections,json
rows=C.load('artifacts/icde_evidence_closure_003/E2/allowed_fitdev_queries.json','AUTHORIZED_FIT_DEV_ONLY')
print(json.dumps({'roles':dict(collections.Counter(x['role'] for x in rows)),'rows':len(rows)}))
