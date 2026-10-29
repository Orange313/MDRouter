"""Reproduction helper: only explicitly hashed corpus shards, never query/qrel archives."""
import argparse,pathlib,json,urllib.request,hashlib,shutil,datetime
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--destination',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();spec=json.loads(pathlib.Path(a.manifest).read_text(encoding='utf8'))
for r in spec['corpus_shards']:
 assert r['path'].startswith('corpus/') and r['path'].endswith('.parquet') and len(r['SHA256'])==64
 if not a.execute:print(json.dumps(r));continue
 d=pathlib.Path(a.destination)/r['repo'].split('/')[-1]/r['revision']/r['path'];d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists(),'No overwrite';assert shutil.disk_usage(d.parent).free>=15*1024**3+r['bytes'];h=hashlib.sha256()
 with urllib.request.urlopen(r['URL'],timeout=60) as response,d.open('xb') as f:
  for b in iter(lambda:response.read(8*1024*1024),b''):f.write(b);h.update(b)
 assert h.hexdigest()==r['SHA256'],str(d)+' hash mismatch: retained for audit'
 print(json.dumps({'path':str(d),'status':'PASS_OFFICIAL_SHA256','time':datetime.datetime.now().astimezone().isoformat()}))
