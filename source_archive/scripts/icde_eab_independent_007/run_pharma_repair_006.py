import common as C
import subprocess,sys,time
from pathlib import Path
gpu=sys.argv[1];logs=C.ROOT/'artifacts/icde_eab_independent_007_logs_001'
for proc in range(3):
    path=logs/f'ColPali_Pharmaceuticals_p{proc}_repair_006.log'
    with path.open('x') as f:r=subprocess.run([sys.executable,str(Path(__file__).with_name('t3_colpali_expansion_006.py')),gpu,'Pharmaceuticals',str(proc)],stdout=f,stderr=subprocess.STDOUT)
    print({'time_bj':C.stamp(),'process':proc,'returncode':r.returncode,'log':str(path),'repair':'literal development role and45-ID assertion only; no scientific parameter change'},flush=True)
    if r.returncode:sys.exit(r.returncode)
    time.sleep(2)
