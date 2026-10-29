from pathlib import Path
import json,collections
root=Path('/path/to/project');base=root/'artifacts/icde_eab_independent_007/T5/replay_003/artifacts'
files={}
for p in [base/'T2/summary_001/input_reads.jsonl',base/'T4/rebuild_004/input_reads.jsonl']:
    for line in p.read_text().splitlines():
        x=json.loads(line);path=x['path'];files[path]=root/path
for path,p in files.items():
    if 'replay_003/reports/T2' in path:continue
    if p.suffix=='.json':
        x=json.loads(p.read_text());print(json.dumps({'path':path,'size':p.stat().st_size,'keys':list(x)[:16],'private_paths':any(z in p.read_text() for z in ['/home/','chengsijia','Orange313','C:\\Users'])}))
print(json.dumps({'paths':[x for x in files if 'replay_003/reports/T2' not in x],'total_bytes':sum(p.stat().st_size for x,p in files.items() if 'replay_003/reports/T2' not in x)},indent=2))
