"""Pinned official acquisition recipe. Never download mixed historical qrels.

Default plan is read-only. Fetches require explicit license acknowledgment,
an exclusive destination, and 15 GiB remaining space. No model training.
"""
import argparse,hashlib,json,shutil,time,urllib.request
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','verify-existing','fetch-biomedical','fetch-models'])
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--root',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--accept-official-license-terms',action='store_true')
    a=p.parse_args();lock=json.loads(a.lock.read_text(encoding='utf8'))
    if a.mode=='plan':print(json.dumps(lock,indent=2));return
    if a.mode=='verify-existing':
        if not a.root:p.error('--root required')
        checks=[{'asset':x['role'],'sha256':sha(a.root/x['project_path']),'expected':x['sha256']} for x in lock['biomedical_files']]
        print(json.dumps({'checks':checks,'PASS':all(x['sha256']==x['expected'] for x in checks)},indent=2))
        if not all(x['sha256']==x['expected'] for x in checks):raise RuntimeError('HASH_MISMATCH')
        return
    if not a.accept_official_license_terms:p.error('Read and explicitly acknowledge the official license terms before downloading')
    if not a.destination:p.error('--destination required')
    a.destination.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(a.destination).free<15*2**30:raise RuntimeError('STORAGE_RESERVE_BELOW_15_GIB')
    if a.mode=='fetch-biomedical':
        for x in lock['biomedical_files']:
            target=a.destination/x['repo_path'];target.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(x['url'],timeout=120) as r,target.open('xb') as f:
                while True:
                    chunk=r.read(8*1024*1024)
                    if not chunk:break
                    if shutil.disk_usage(a.destination).free<15*2**30:raise RuntimeError('STORAGE_RESERVE_BELOW_15_GIB_PARTIAL_PRESERVED')
                    f.write(chunk)
            if sha(target)!=x['sha256']:raise RuntimeError('HASH_MISMATCH_PARTIAL_PRESERVED '+str(target))
            print(json.dumps({'verified':x['role'],'sha256':x['sha256']}),flush=True)
    else:
        from huggingface_hub import snapshot_download
        for x in lock['models']:
            if shutil.disk_usage(a.destination).free<15*2**30:raise RuntimeError('STORAGE_RESERVE_BELOW_15_GIB')
            snapshot_download(repo_id=x['repo'],revision=x['revision'],cache_dir=str(a.destination/'hf-cache'))
            print(json.dumps({'model':x['repo'],'revision':x['revision'],'note':'verify exact consumed files against the run model-asset ledger before execution'}),flush=True)
    (a.destination/'acquisition_receipt.json').write_text(json.dumps({'status':'PASS_PINNED_ACQUISITION','time_unix':time.time(),'lock_sha256':sha(a.lock),'mode':a.mode},indent=2),encoding='utf8')

if __name__=='__main__':main()
