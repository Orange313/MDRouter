"""CPU-only replay from immutable per-query outcomes and timing traces; no new evaluation."""
import common as C
import argparse,runpy,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=False)
original_A=C.ART;original_R=C.REPORT
C.ART=dest/'artifacts';C.REPORT=dest/'reports';C.ART.mkdir();C.REPORT.mkdir()
for name in ['T1','T1_preparation']:(C.ART/name).symlink_to(original_A/name,target_is_directory=True)
(C.ART/'T2').mkdir();(C.REPORT/'T2').mkdir()
for i in range(3):
    (C.ART/f'T2/process_{i}_001').symlink_to(original_A/f'T2/process_{i}_001',target_is_directory=True)
    (C.REPORT/f'T2/process_{i}_001').symlink_to(original_R/f'T2/process_{i}_001',target_is_directory=True)
if (original_R/'correctness_supplement').exists():(C.REPORT/'correctness_supplement').symlink_to(original_R/'correctness_supplement',target_is_directory=True)
(C.REPORT/'T3').symlink_to(original_R/'T3',target_is_directory=True)
runpy.run_path(str(Path(__file__).with_name('t2_summarize_001.py')),run_name='__main__')
checks=[]
for a in (original_R/'T2/summary_001').glob('*.csv'):
    b=C.REPORT/'T2/summary_001'/a.name;checks.append({'table':a.name,'original_SHA256':C.sha(a),'replay_SHA256':C.sha(b),'identical':C.sha(a)==C.sha(b)})
C.INPUT={};runpy.run_path(str(Path(__file__).with_name('t4_assemble_003.py')),run_name='__main__')
for a in (original_R/'T4').glob('*.csv'):
    b=C.REPORT/'T4'/a.name;checks.append({'table':'T4/'+a.name,'original_SHA256':C.sha(a),'replay_SHA256':C.sha(b),'identical':C.sha(a)==C.sha(b)})
C.write(dest/'replay_result.json',{'status':'PASS' if all(x['identical'] for x in checks) else 'FAIL','scope':'CPU_NUMERIC_REPLAY_ONLY_NOT_NEW_SAMPLES','checks':checks})
assert all(x['identical'] for x in checks),'NUMERIC_REPLAY_MISMATCH'
print(json.dumps({'status':'PASS_CPU_REPLAY','tables':len(checks),'new_GPU_queries':0,'output':str(dest)}))
