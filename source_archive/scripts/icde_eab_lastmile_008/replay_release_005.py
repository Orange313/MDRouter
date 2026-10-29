from pathlib import Path
import json,shutil,subprocess,time
R=Path('/path/to/project');S=R/'artifacts/icde_eab_lastmile_008/L2';D=S/'release_candidate_005'
shutil.copytree(S/'release_candidate_004',D,ignore=shutil.ignore_patterns('.git','__pycache__'))
(D/'engineering_failure_ledger.json').write_text(json.dumps({'candidate001':'environment metadata lookup failed; original retained','clean_checkout001':'required small index metadata was excluded by broad ignore rule; original failed replay retained; root-anchored rule fixed in002','candidate003':'large detailed index traces caused210MB snapshots; preserved privately; candidate004 uses scalar profiles plus trace hash inventory','candidate004':'server Git lacks init --initial-branch; failure before commit or upload;005 uses compatible git init then checkout -b','GPU_process2_001':'pre-CUDA idle check reported lingering100percent utilization; retry002 preserved new directory and passed'},indent=2))
def g(a,cwd=D):return subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
g(['init','--quiet']);g(['checkout','--quiet','-b','codex/lastmile-artifact-008']);g(['config','user.name','Artifact Maintainers']);g(['config','user.email','artifact@example.invalid']);g(['add','.'])
m=json.loads((D/'input_manifest.json').read_text());tracked=set(g(['ls-files']).splitlines());assert all(x['file'] in tracked for x in m.values())
g(['commit','--quiet','-m','Prepare size-reviewed EAB artifact: exact numerical replay and guarded execution evidence'])
C=S/'clean_checkout_005';subprocess.run(['git','clone','--quiet',str(D),str(C)],check=True)
assert not any(p.is_symlink() for p in C.rglob('*'))
t=time.perf_counter()
with (S/'clean_replay_005.log').open('x') as f:r=subprocess.run([str(S/'clean_environment_001/bin/python'),str(C/'reproduce.py'),'--output',str(S/'clean_replay_005')],cwd=C,stdout=f,stderr=subprocess.STDOUT)
receipt={'status':'PASS' if r.returncode==0 else 'FAIL','candidate_commit':g(['rev-parse','HEAD']),'returncode':r.returncode,'wall_seconds':time.perf_counter()-t,'original_checkout_imported':False,'public_upload':'NOT_RUN_PENDING_USER_APPROVAL'}
(S/'clean_checkout_005_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);assert r.returncode==0
