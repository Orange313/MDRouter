"""Private, isolated Git snapshot; never changes the shared project repository."""
import pathlib,shutil,subprocess,json,hashlib
def run(root,run_id,dest):
 root=pathlib.Path(root);dest=pathlib.Path(dest);repo=dest/'code_repository_001';repo.mkdir(exist_ok=False)
 paths=list((root/'scripts'/run_id).glob('*.py'))+[root/'scripts/cross_domain_four_plan_001.py',root/'configs/hr_pilot_sparse_004.json',root/'manifests/hr_models_verified.json']
 for folder in ['src/aaber','src/mdrouter_v5']:paths+=list((root/folder).rglob('*.py'))
 for p in paths:
  out=repo/p.relative_to(root);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
 def git(args):return subprocess.check_output(['git','-C',str(repo)]+args,text=True).strip()
 git(['init','--quiet']);git(['symbolic-ref','HEAD','refs/heads/codex/icde-evidence-closure-003']);git(['add','.']);git(['-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','--quiet','-m','Archive evidence closure code and frozen backend dependencies']);commit=git(['rev-parse','HEAD']);bundle=dest/'code_snapshot_001.bundle';git(['bundle','create',str(bundle),'--all']);info={'code_snapshot_commit':commit,'shared_project_HEAD_unchanged':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'bundle_path':str(bundle),'bundle_SHA256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'snapshot_file_count':len(paths),'source_file_SHA256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'publication':'PRIVATE_LOCAL_ONLY_NO_EXTERNAL_PUSH'}
 (dest/'code_commit_record.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8');return info
