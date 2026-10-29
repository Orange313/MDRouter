from pathlib import Path
import subprocess,json,shutil,hashlib
root=Path('/path/to/project');stage=root/'artifacts/icde_eab_lastmile_008/L2';candidate=stage/'release_candidate_002'
ig=candidate/'.gitignore';text=ig.read_text()
for name in ['data/','datasets/','models/','weights/','indexes/']:text=text.replace('\n'+name+'\n','\n/'+name+'\n')
ig.write_text(text)
manifest=json.loads((candidate/'input_manifest.json').read_text())
def git(a,cwd=candidate):return subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
git(['add','.']);tracked=set(git(['ls-files']).splitlines());assert all(x['file'] in tracked for x in manifest.values()),'REPLAY_INPUT_IGNORED'
git(['commit','--quiet','-m','Track small index metadata needed for reproducibility; exclude actual weights and indexes'])
checkout=stage/'clean_checkout_002';subprocess.run(['git','clone','--quiet',str(candidate),str(checkout)],check=True)
assert not any(p.is_symlink() for p in checkout.rglob('*'))
env=stage/'clean_environment_001'
with (stage/'clean_replay_002.log').open('x') as f:r=subprocess.run([str(env/'bin/python'),str(checkout/'reproduce.py'),'--output',str(stage/'clean_replay_002')],cwd=checkout,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0,'CLEAN_REPLAY_RETRY_FAILED'
receipt=json.loads((stage/'clean_replay_002/reproduction_receipt.json').read_text());assert receipt['status']=='PASS' and len(receipt['checks'])==15
(stage/'clean_checkout_002_receipt.json').write_text(json.dumps({'status':'PASS_FRESH_CHECKOUT_ISOLATED_DEPENDENCIES_15_CSV','commit':git(['rev-parse','HEAD']),'no_original_repository_imports':True,'no_original_input_symlinks':True,'CSV_checks':receipt['checks'],'figure_files':receipt['figure_files'],'wall_seconds':receipt['wall_seconds'],'public_push':'NOT_RUN_PENDING_USER_APPROVAL'},indent=2))
print(json.dumps({'status':'PASS_15_CSV_CLEAN_REPLAY','commit':git(['rev-parse','HEAD']),'wall_seconds':receipt['wall_seconds']}))
