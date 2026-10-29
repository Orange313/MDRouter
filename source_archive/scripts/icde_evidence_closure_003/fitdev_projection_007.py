import pathlib,json,hashlib,urllib.parse,subprocess,datetime,concurrent.futures
BASE=pathlib.Path(__file__).resolve().parents[1];OUT=BASE/'public_sources/official_fitdev_projection_007';OUT.mkdir(exist_ok=False)
REV='262e203d7c59c55947f7042812e0bcc1eee190b1';REPO='vidore/vidore_v3_pharmaceuticals'
audit=BASE.parent/'deliveries/pharma_fixedcorpus_split_audit_001_download_001/内容/artifacts/pharma_fixedcorpus_split_audit_001/audit_001';qs=[]
for name in ['fit_unlabeled_queries.json','development_unlabeled_queries.json']:qs+=json.loads((audit/name).read_text(encoding='utf8'))
ids=sorted(int(q['query_id']) for q in qs);assert len(set(ids))==len(ids)==126
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def get(url,stem):
 for attempt in range(1,4):
  p=OUT/f'{stem}_attempt_{attempt}.json';r=subprocess.run(['curl.exe','-L','--connect-timeout','8','--max-time','40','-sS','-D',str(p)+'.headers','-o',str(p),'-w','%{http_code}',url],capture_output=True,text=True)
  (OUT/f'{stem}_attempt_{attempt}.receipt.json').write_text(json.dumps({'time_bj':now(),'url':url,'code':r.returncode,'HTTP':r.stdout,'stderr':r.stderr,'sha256':sha(p) if p.exists() else None}),encoding='utf8')
  if r.returncode==0 and r.stdout=='200':return json.loads(p.read_text(encoding='utf8')),p
 raise RuntimeError('Public request failed, preserved: '+stem)
def origin(stem):
 a,p=get(f'https://huggingface.co/api/datasets/{REPO}?blobs=true',stem);q=next(x for x in a['siblings'] if x['rfilename']=='qrels/test-00000-of-00001.parquet');assert q['lfs']['sha256']=='d2812c5b425d520fb4f466d2c0aa0ce811e493bf85cf9a576c87cd48929882bf';return {'revision':a['sha'],'metadata_sha256':sha(p),'fixed_source_revision':REV,'qrels_byte_identical':True}
before=origin('origin_before')
def one(qid):
 for folder in ['official_fitdev_projection_003','official_fitdev_projection_004','official_fitdev_projection_005','official_fitdev_projection_006']:
  for p in (BASE/'public_sources'/folder).glob(f'allowed_q{qid}_offset0_attempt_*.json'):
   try:
    d=json.loads(p.read_text(encoding='utf8'))
    if 'rows' in d and d.get('partial') is False and d['num_rows_total']==len(d['rows']):
     rr=[x['row'] for x in d['rows']]
     if rr and all(int(r['query_id'])==qid for r in rr):return rr,[{'query_id':qid,'reuse_projection_response':str(p),'response_sha256':sha(p),'returned_rows':len(rr)}]
   except (ValueError,KeyError):pass
 rows=[];offset=0;receipts=[]
 while True:
  url='https://datasets-server.huggingface.co/filter?'+urllib.parse.urlencode({'dataset':REPO,'config':'qrels','split':'test','where':f'"query_id">={qid} AND "query_id"<={qid}','offset':offset,'length':100});data,p=get(url,f'allowed_q{qid}_offset{offset}')
  assert data.get('partial') is False;assert not any(x.get('truncated_cells') for x in data['rows']);rr=[x['row'] for x in data['rows']];assert all(int(r['query_id'])==qid for r in rr);rows+=rr;receipts.append({'query_id':qid,'url':url,'response_sha256':sha(p),'returned_rows':len(rr)});offset+=len(rr)
  if offset>=data['num_rows_total']:break
  assert rr
 assert rows
 return rows,receipts
allrows=[];receipts=[];failures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 futures={ex.submit(one,q):q for q in ids}
 for n,f in enumerate(concurrent.futures.as_completed(futures),1):
  try:r,s=f.result();allrows+=r;receipts+=s
  except Exception as e:failures.append({'query_id':futures[f],'error':str(e)})
  if n%10==0:print(json.dumps({'completed_queries':n,'allowed_total':126,'failed':len(failures),'time_bj':now()}),flush=True)
after=origin('origin_after');(OUT/'failures.json').write_text(json.dumps(failures),encoding='utf8');assert not failures,failures
assert set(int(r['query_id']) for r in allrows)==set(ids);assert len({(r['query_id'],r['corpus_id']) for r in allrows})==len(allrows);assert all(r['score'] in [1,2] for r in allrows)
rows=sorted([{k:r[k] for k in ['query_id','corpus_id','score']} for r in allrows],key=lambda r:(r['query_id'],r['corpus_id']));p=OUT/'fitdev_qrels.json';p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');m={'time_bj':now(),'revision':REV,'origin_before':before,'origin_after':after,'scope':'only allowed frozen Fit81/Dev45 query IDs via official per-ID HTTP filter; raw mixed local qrels never opened','allowed_ids':ids,'rows':len(rows),'requests':receipts,'fitdev_qrels_SHA256':sha(p),'Cal_Eval_records_delivered':0};(OUT/'projection_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'status':'PASS_OFFICIAL_FILTERED_PROJECTION','rows':len(rows),'queries':126,'SHA256':sha(p),'time_bj':now()}),flush=True)
