"""Run on local Windows host. Public corpus or allowlisted remote qrels projection only."""
import pathlib,subprocess,json,hashlib,urllib.parse,datetime,sys,time
BASE=pathlib.Path(__file__).resolve().parents[1];OUT=BASE/'public_sources';OUT.mkdir(exist_ok=True)
REV='262e203d7c59c55947f7042812e0bcc1eee190b1';REPO='vidore/vidore_v3_pharmaceuticals'
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def fetch(url,p,timeout=120):
 assert not p.exists(),p
 result=subprocess.run(['curl.exe','-L','--connect-timeout','10','--max-time',str(timeout),'-sS','-D',str(p)+'.headers','-o',str(p),'-w','%{http_code}',url],capture_output=True,text=True)
 (p.parent/(p.name+'.request.json')).write_text(json.dumps({'time_bj':stamp(),'url':url,'returncode':result.returncode,'HTTP':result.stdout,'stderr':result.stderr,'sha256':sha(p) if p.exists() else None},ensure_ascii=False,indent=2),encoding='utf8')
 if result.returncode or result.stdout!='200':raise RuntimeError((p.name,result.returncode,result.stdout,result.stderr))
 return json.loads(p.read_text(encoding='utf8')) if p.suffix=='.json' else None
if sys.argv[1]=='corpus':
 targets=[('test-00000-of-00002.parquet',385083648,'0f51d221a1def60c6d1436a31507e69be42a6a1da4a9ae45ff6e231751466ac5'),('test-00001-of-00002.parquet',366386229,'827305c58164c9e69581953560b1e76349dbc86a5a06df91d16c92023175e468')]
 for name,size,digest in targets:
  p=OUT/name;fetch(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/corpus/{name}',p,1800);assert p.stat().st_size==size and sha(p)==digest;print(json.dumps({'corpus':name,'bytes':size,'SHA256':digest,'time_bj':stamp(),'status':'PASS'}),flush=True)
elif sys.argv[1]=='qrels':
 pdir=OUT/'official_fitdev_projection_001';pdir.mkdir(exist_ok=False)
 audit=BASE.parents[0]/'deliveries/pharma_fixedcorpus_split_audit_001_download_001/内容/artifacts/pharma_fixedcorpus_split_audit_001/audit_001'
 queries=[]
 for name in ['fit_unlabeled_queries.json','development_unlabeled_queries.json']:queries+=json.loads((audit/name).read_text(encoding='utf8'))
 ids=sorted(int(q['query_id']) for q in queries);assert len(ids)==126 and len(set(ids))==126
 before=fetch(f'https://huggingface.co/api/datasets/{REPO}?blobs=true',pdir/'origin_before.json')
 def verify(api):
  assert api['sha']==REV
  q=next(x for x in api['siblings'] if x['rfilename']=='qrels/test-00000-of-00001.parquet');assert q['lfs']['sha256']=='d2812c5b425d520fb4f466d2c0aa0ce811e493bf85cf9a576c87cd48929882bf'
 verify(before);allrows=[];requests=[]
 for chunk,start in enumerate(range(0,len(ids),21)):
  allow=set(ids[start:start+21]);where=' OR '.join(f'"query_id"=\'{i}\'' for i in sorted(allow));offset=0
  while True:
   params={'dataset':REPO,'config':'qrels','split':'test','where':where,'offset':offset,'length':100};url='https://datasets-server.huggingface.co/filter?'+urllib.parse.urlencode(params);p=pdir/f'allowed_chunk_{chunk:02d}_offset_{offset:05d}.json';data=fetch(url,p,180)
   assert data.get('partial') is False and not any(x.get('truncated_cells') for x in data['rows'])
   rows=[x['row'] for x in data['rows']];assert all(int(r['query_id']) in allow for r in rows),'Projection scope violated; stop'
   allrows.extend(rows);requests.append({'url':url,'sha256':sha(p),'returned_rows':len(rows),'allowlist':sorted(allow)});offset+=len(rows)
   if offset>=data['num_rows_total']:break
   assert rows,'Pagination stalled'
  print(json.dumps({'projection_chunk':chunk,'allowed_queries':len(allow),'returned_rows_so_far':len(allrows),'time_bj':stamp()}),flush=True)
 after=fetch(f'https://huggingface.co/api/datasets/{REPO}?blobs=true',pdir/'origin_after.json');verify(after)
 assert set(int(r['query_id']) for r in allrows)==set(ids)
 tuples=[(int(r['query_id']),int(r['corpus_id'])) for r in allrows];assert len(tuples)==len(set(tuples));assert all(r['score'] in [1,2] for r in allrows)
 clean=[{k:r[k] for k in ['query_id','corpus_id','score']} for r in allrows];(pdir/'fitdev_qrels.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding='utf8')
 (pdir/'projection_manifest.json').write_text(json.dumps({'time_bj':stamp(),'revision':REV,'official_qrels_lfs_sha256':'d2812c5b425d520fb4f466d2c0aa0ce811e493bf85cf9a576c87cd48929882bf','scope':'only frozen Fit81+Dev45 official rows delivered; no raw mixed local qrels open','allowed_ids':ids,'requests':requests,'rows':len(clean),'fitdev_qrels_SHA256':sha(pdir/'fitdev_qrels.json'),'Cal_Eval_records_delivered':0},ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':'PASS_OFFICIAL_FILTERED_PROJECTION','rows':len(clean),'queries':len(ids),'time_bj':stamp()}),flush=True)
