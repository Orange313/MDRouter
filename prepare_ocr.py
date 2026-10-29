from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,subprocess,time
import pyarrow.parquet as pq
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--tesseract',default='tesseract');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=pq.read_table(a.corpus).to_pylist();start=time.perf_counter()
version=subprocess.check_output([a.tesseract,'--version'],stderr=subprocess.STDOUT,text=True)
def one(r):
    q=out/(str(r['corpus-id'])+'.png');Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB').save(q)
    z=subprocess.run([a.tesseract,str(q),'stdout','-l','eng','--psm','3'],capture_output=True,text=True,check=True)
    return {'corpus_id':int(r['corpus-id']),'doc_id':r['doc-id'],'markdown':z.stdout,'image_SHA256':hashlib.sha256(r['image']['bytes']).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:data=sorted(pool.map(one,rows),key=lambda x:x['corpus_id'])
(out/'pages.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));(out/'receipt.json').write_text(json.dumps({'pages':len(data),'wall_s':time.perf_counter()-start,'version':version,'qrels_read':False,'note':'Portable recipe, same eng/psm3; exact OCR needs original runtime hashes. Not a measured benchmark until executed.'},indent=2))
