from pathlib import Path
import hashlib, json, shutil, subprocess
def run(root,run,dest):
    dest.mkdir(parents=True,exist_ok=False);repo=dest/'code_repository';repo.mkdir()
    paths=list((root/'scripts'/run).glob('*.py'))
    for folder in ['src/aaber','src/mdrouter_v5']:
        paths+=list((root/folder).rglob('*.py'))
    for name in ['common.py','engine_001.py','learning_base_001.py','frontier_math.py','stream_worker_v3_001.py']:
        paths.append(root/'scripts/icde_eab_validation_005'/name)
    paths+=[root/'scripts/cross_domain_four_plan_001.py',root/'manifests/hr_models_verified.json']
    for p in paths:
        q=repo/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    def git(args):return subprocess.check_output(['git','-C',str(repo)]+args,text=True).strip()
    git(['init','--quiet']);git(['symbolic-ref','HEAD','refs/heads/codex/icde-eab-lastmile-008']);git(['add','.'])
    git(['-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','--quiet','-m','Version guarded EAB experiment source'])
    bundle=dest/'source.bundle';git(['bundle','create',str(bundle),'--all'])
    d={'code_snapshot_commit':git(['rev-parse','HEAD']),'base_project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'bundle_SHA256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'source_file_SHA256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (dest/'commit.json').write_text(json.dumps(d,indent=2),encoding='utf8');return d
