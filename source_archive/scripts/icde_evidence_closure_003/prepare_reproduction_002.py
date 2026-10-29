"""Prepare a new isolated run namespace; execute only when explicitly invoked with --execute."""
import pathlib,argparse,re,shutil,json,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--root',default='/path/to/project');p.add_argument('--run-id',required=True);p.add_argument('--gpus',default='0,4,5');p.add_argument('--execute',action='store_true');a=p.parse_args();root=pathlib.Path(a.root).resolve();assert re.fullmatch(r'icde_evidence_closure_003_reproduction_[a-z0-9_]+',a.run_id);gpus=a.gpus.split(',');assert len(gpus)==3 and len(set(gpus))==3 and all(g.isdigit() for g in gpus)
old='icde_evidence_closure_003';scripts=root/'scripts'/a.run_id;scripts.mkdir(exist_ok=False)
for src in pathlib.Path(__file__).parent.glob('*.py'):
 text=src.read_text(encoding='utf8')
 if src.name=='common.py':text=text.replace("RUN='icde_evidence_closure_003'",f"RUN='{a.run_id}'").replace("ROOT=pathlib.Path('/path/to/project')",f'ROOT=pathlib.Path({str(root)!r})')
 (scripts/src.name).write_text(text,encoding='utf8')
commands=[['e0_001.py'],['e1_driver_001.py']+gpus,['e1_analyze_001.py'],['e2_preflight_001.py'],['e2_outcomes_001.py',gpus[0]],['e2_learning_001.py'],['e2_uncertainty_001.py'],['e2_closure_seal_001.py'],['pharma_components_001.py',gpus[0]],['e3_001.py'],['e4_freeze_001.py'],['feature_semantics_audit_002.py'],['e4_driver_001.py'],['e4_analyze_001.py'],['queue_noise_diagnostic_001.py'],['e5_reproduction_003.py']];plan={'new_run_id':a.run_id,'scope':'new output directories; old artifacts immutable; no Orange writes; no Cal/Eval/oldBL/test/CS access','commands':commands,'Pharma_index':'reuse only verified completed fixed index; no index overwrite','status':'PREPARED_NOT_EXECUTED' if not a.execute else 'REPRODUCTION_EXECUTION_REQUESTED'};(scripts/'reproduction_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
if not a.execute:print(json.dumps(plan));sys.exit(0)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1');python=str(root/'.venv/bin/python')
for cmd in commands:
 if cmd[0]=='e2_outcomes_001.py':
  dest=root/'artifacts'/a.run_id/'E2/public_projection_final_001';shutil.copytree(root/'artifacts'/old/'E2/public_projection_final_001',dest)
  # Existing fixed shared index is immutable and verified from its original build manifest.
  import hashlib
  index=root/'cache/indexes/pharma_fixedcorpus_index_001';manifest=json.loads((index/'index_manifest.json').read_text());assert hashlib.sha256((index/'index_manifest.json').read_bytes()).hexdigest()=='6c280d2a1e4806e10af5e3eeeedd45fb295000811bcf3dbd1d2e1826b2760920'
  for rel,h in manifest['files'].items():assert hashlib.sha256((index/rel).read_bytes()).hexdigest()==h
 runenv=dict(env,CUDA_VISIBLE_DEVICES=cmd[1]) if cmd[0] in ['e2_outcomes_001.py','pharma_components_001.py'] else env
 subprocess.run([python,str(scripts/cmd[0])]+cmd[1:],cwd=root,env=runenv,check=True)
print('REPRODUCTION_COMPLETED_NO_EXTERNAL_WRITES')
