"""Replay all38 numeric CSVs and5 figure families from a clean checkout."""
from pathlib import Path
import argparse,json,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent;t=time.perf_counter()
for name,script in [('base',root/'reproduce.py'),('cost009',root/'cost009/reproduce.py')]:
    subprocess.run([sys.executable,str(script),'--output',str(out/name)],check=True,cwd=root)
base=json.loads((out/'base/reproduction_receipt.json').read_text());cost=json.loads((out/'cost009/reproduction_receipt.json').read_text())
# Merge already verified materializations into one read-only logical input namespace.
import os,runpy,hashlib
ws=out/'lifecycle010_workspace';ws.mkdir()
for prefix in ['reports','artifacts']:
    dst=ws/prefix;dst.mkdir()
    for stage in ['base','cost009']:
        source=out/stage/'workspace'/prefix
        if source.exists():
            for item in source.iterdir():
                if not (dst/item.name).exists():(dst/item.name).symlink_to(item,target_is_directory=item.is_dir())
os.environ['MDROUTER_REPLAY_ROOT']=str(ws);sys.path.insert(0,str(root/'cost009/repro'));import common as C
C.RUN='icde_eab_release_validation_010';C.ROOT=ws;C.ART=ws/'artifacts'/C.RUN;C.REPORT=ws/'reports'/C.RUN
runpy.run_path(str(root/'repro/lifecycle010_portable.py'),run_name='__main__')
lifecycle=[]
for name,h in json.loads((root/'expected_lifecycle010_hashes.json').read_text()).items():
    actual=hashlib.sha256((C.REPORT/'A0/replay_002'/name).read_bytes()).hexdigest();lifecycle.append({'file':name,'expected':h,'actual':actual,'identical':h==actual})
assert all(x['identical'] for x in lifecycle)
receipt={'status':'PASS','lifecycle010':lifecycle,'base':base,'cost009':cost,'CSV_expected':38,'wall_seconds':time.perf_counter()-t,'GPU_executions':0,'scope':'archived numerical CPU replay, not new hardware measurements, independent evaluation, or public hosting'}
(out/'all38_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PASS_ALL38','wall_seconds':receipt['wall_seconds']}))
