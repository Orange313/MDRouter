import common as C
import subprocess,sys,time
from pathlib import Path
gpu=sys.argv[1];out=C.ROOT/'artifacts/icde_eab_independent_007_logs_001';out.mkdir(exist_ok=True)
assert (C.ART/'T1/manifest.json').exists()
for proc in range(3):
    log=out/f'biomedical_process_{proc}_001.log'
    with log.open('x') as f:
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('t2_biomedical_exact_004.py')),gpu,str(proc)],stdout=f,stderr=subprocess.STDOUT)
    print({'time_bj':C.stamp(),'process':proc,'returncode':p.returncode,'log':str(log)},flush=True)
    if p.returncode:sys.exit(p.returncode)
    time.sleep(2)
