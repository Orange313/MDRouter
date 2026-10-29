"""Fresh-checkout CPU numerical reproduction: no private workspace dependency."""
from pathlib import Path
import argparse,json,gzip,hashlib,os,sys,runpy,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);ws=out/'workspace';ws.mkdir();start=time.perf_counter()
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((base/'input_manifest.json').read_text())
for target,x in manifest.items():
    if target.startswith(('reports/icde_eab_lastmile_008/L0/','reports/icde_eab_lastmile_008/L1/summary_001/')):continue
    data=gzip.decompress((base/x['file']).read_bytes());assert sha(data)==x['sha256'];q=ws/target;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
os.environ['MDROUTER_REPLAY_ROOT']=str(ws);sys.path.insert(0,str(base/'repro'))
for script in ['t2_summarize_001.py','t4_assemble_004.py']:runpy.run_path(str(base/'repro'/script),run_name='__main__')
# Replay008 descriptive diagnostics, with a separate immutable run namespace.
import common as C
C.RUN='icde_eab_lastmile_008';C.ART=ws/'artifacts'/C.RUN;C.REPORT=ws/'reports'/C.RUN;C.SEED=202610281450;C.INPUT={};C.A=C.R=None
for script in ['l0_paired_portable.py','l1_summary_portable.py']:runpy.run_path(str(base/'repro'/script),run_name='__main__')
checks=[]
for name,h in {**json.loads((base/'expected_numeric_hashes.json').read_text()),**json.loads((base/'expected_extension_hashes.json').read_text())}.items():
    actual=sha((ws/name).read_bytes());checks.append({'table':name,'expected':h,'actual':actual,'identical':actual==h})
receipt={'status':'PASS' if all(x['identical'] for x in checks) else 'FAIL','checks':checks,'wall_seconds':time.perf_counter()-start,'original_research_checkout_required':False,'network_required':False,'new_queries':0,'new_GPU_work':0,'figure_files':[str(x.relative_to(out)) for x in (ws/'reports/icde_eab_independent_007/T4/rebuild_004/figures').glob('*')]}
(out/'reproduction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'CSV_count':len(checks),'wall_seconds':receipt['wall_seconds'],'figures':len(receipt['figure_files'])}));assert receipt['status']=='PASS'
