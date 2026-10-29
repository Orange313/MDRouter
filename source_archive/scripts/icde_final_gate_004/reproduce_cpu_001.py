"""Recompute numerical tables from immutable raw allowed data into a NEW tree."""
import common as C
import pathlib,sys,runpy,shutil,json
mode,outarg=sys.argv[1:];out=pathlib.Path(outarg);out=out if out.is_absolute() else C.ROOT/out;out.mkdir(parents=True,exist_ok=False);source=pathlib.Path(__file__).parent
if mode in ['resident','learning']:
 def fresh_start(stage):
  C.STAGE='REPRO_'+stage;C.A=out/'artifacts'/stage;C.R=out/'reports'/stage;C.A.mkdir(parents=True,exist_ok=False);C.R.mkdir(parents=True,exist_ok=False);sys.addaudithook(C.protected);C.write(C.A/'reproduction_start.json',{'time_bj':C.stamp(),'mode':mode,'inputs_root':str(C.ART),'new_output_root':str(out)})
 C.start=fresh_start;runpy.run_path(str(source/('resident_analyze_002.py' if mode=='resident' else 's3_prepare_001.py')),run_name='__main__')
elif mode=='AA':
 # Copy only closed raw request/cell/guard/telemetry metadata; no GPU/model rebuild or sealed data.
 original=C.ART;C.ART=out/'artifacts';C.REPORT=out/'reports';ap=C.ART/'S1';rp=C.REPORT/'S1';ap.mkdir(parents=True);rp.mkdir(parents=True)
 for p in (original/'S1/workers').rglob('*.json'):
  if p.name not in ['request_rows.json','cell_rows.json','resource_guard.json','telemetry.json','stage_start.json','failure.json']:continue
  dst=ap/p.relative_to(original/'S1');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
 runpy.run_path(str(source/'aa_full_analysis_005.py'),run_name='__main__')
else:raise ValueError('mode must be resident, learning, AA; hardware replay has separate protocol, not executed by CPU reproducer')
files={str(p.relative_to(out)):C.sha(p) for p in out.rglob('*') if p.is_file()};C.write(out/'reproduction_manifest.json',{'time_bj':C.stamp(),'mode':mode,'source_code_SHA256':{p.name:C.sha(p) for p in source.glob('*.py')},'outputs':files,'status':'PASS_RECOMPUTED_CPU_ONLY','sealed_reads':0,'new_independent_samples':0})
