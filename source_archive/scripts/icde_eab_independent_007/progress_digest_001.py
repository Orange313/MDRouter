import common as C
import json,csv,collections,subprocess,shutil
from pathlib import Path
out={'time_bj':C.stamp(),'pilot':[],'disk_free_GiB':shutil.disk_usage(C.ROOT).free/2**30}
for d in ['vidore_v3_hr','vidore_v3_industrial']:
    for p in range(3):
        a=C.ART/f'T3/pilot_{d}_p{p}_002/completed.json'
        if a.exists():
            done=json.loads(a.read_text());raw=list(csv.DictReader((C.REPORT/f'T3/pilot_{d}_p{p}_002/requests.csv').open()));groups=collections.defaultdict(list)
            for r in raw:groups[r['plan']].append(r)
            out['pilot'].append({'domain':d,'process':p,'status':done['status'],'requests':len(raw),'metrics':{k:{m:sum(float(r[m]) for r in rs)/len(rs) for m in ['nDCG10','r','E2E_s']} for k,rs in groups.items()}})
out['GPU_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
print(json.dumps(out,ensure_ascii=False))
