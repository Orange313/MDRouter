"""Sequential independent processes. No occupancy reservation, no retries hidden."""
from pathlib import Path
import sys,subprocess,time,json
root=Path('PROJECT_ROOT');run='icde_eab_lastmile_008';logs=root/'artifacts/icde_eab_lastmile_008_operations_001';logs.mkdir(exist_ok=False)
for proc in range(3):
    with (logs/f'L1_p{proc}.log').open('x') as f:
        r=subprocess.run([sys.executable,str(root/'scripts'/run/'l1_biomedical_colpali_001.py'),'3',str(proc)],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'process':proc,'exit_code':r.returncode,'time_unix':time.time()}),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
(logs/'L1_all_completed.json').write_text(json.dumps({'status':'PASS_3_PROCESSES_EXECUTED','time_unix':time.time()}))
