import common as C
import subprocess,sys
from pathlib import Path
gpu=sys.argv[1];logs=C.ROOT/'artifacts/icde_eab_independent_007_logs_001'
for d in C.DOM:
    path=logs/f'primary_parity_{d}_001.log'
    with path.open('x') as f:r=subprocess.run([sys.executable,str(Path(__file__).with_name('primary_scorer_parity_001.py')),gpu,d],stdout=f,stderr=subprocess.STDOUT)
    print({'time_bj':C.stamp(),'domain':d,'returncode':r.returncode,'role':'EXPOSED_DEV_CORRECTNESS_ONLY_NOT_NEW_BENCHMARK','log':str(path)},flush=True)
    if r.returncode:sys.exit(r.returncode)
