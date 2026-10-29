from pathlib import Path
import argparse,gzip,hashlib,json,os,runpy,sys,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();base=Path(__file__).resolve().parent;out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);ws=out/'workspace';ws.mkdir();start=time.perf_counter();sha=lambda b:hashlib.sha256(b).hexdigest()
for target,x in json.loads((base/'input_manifest.json').read_text()).items():
    b=gzip.decompress((base/x['file']).read_bytes());assert sha(b)==x['sha256']
    if target.startswith('reports/icde_eab_submission_gate_009/Y1/summary_001/'):continue
    q=ws/target;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
os.environ['MDROUTER_REPLAY_ROOT']=str(ws);sys.path.insert(0,str(base/'repro'));runpy.run_path(str(base/'repro/y1_summary_portable.py'),run_name='__main__');checks=[]
for n,h in json.loads((base/'expected_hashes.json').read_text()).items():
    actual=sha((ws/n).read_bytes());checks.append({'file':n,'expected':h,'actual':actual,'identical':actual==h})
r={'status':'PASS' if all(x['identical'] for x in checks) else 'FAIL','tables':len(checks),'checks':checks,'wall_s':time.perf_counter()-start,'network':False,'GPU':False};(out/'reproduction_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'tables':len(checks),'wall_s':r['wall_s']}));assert r['status']=='PASS'
