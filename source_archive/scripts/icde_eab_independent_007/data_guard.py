"""Default-deny acquisition boundary; no release authority in this run."""
from pathlib import Path
import datetime, json
class Guard:
    def __init__(self,ledger):self.ledger=Path(ledger)
    def read(self,path,role,allowed_paths=()):
        p=Path(path).resolve();permitted=role in {'DEV','FIT','UNLABELLED_ASSET','METADATA'} and str(p) in {str(Path(x).resolve()) for x in allowed_paths}
        self.ledger.parent.mkdir(parents=True,exist_ok=True)
        with self.ledger.open('a',encoding='utf8') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(p),'role':role,'decision':'ALLOW' if permitted else 'DENY_BEFORE_OPEN'})+'\n')
        if not permitted:raise PermissionError('ROLE/PATH NOT AUTHORIZED; payload not opened')
        return p.read_bytes()
