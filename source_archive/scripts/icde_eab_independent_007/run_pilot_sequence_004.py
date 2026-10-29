import subprocess,sys,time,json
from pathlib import Path
import common as C
gpu,domain=sys.argv[1:3]
out=C.ROOT/'artifacts/icde_eab_independent_007_logs_001';out.mkdir(exist_ok=True)
for proc in range(3):
    log=out/f'colpali_{domain}_p{proc}_004.log'
    with log.open('x') as f:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('t3_colpali_pilot_004.py')),gpu,domain,str(proc)],stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'time_bj':C.stamp(),'domain':domain,'process':proc,'returncode':result.returncode,'log':str(log)}),flush=True)
    if result.returncode:sys.exit(result.returncode)
    time.sleep(2)
