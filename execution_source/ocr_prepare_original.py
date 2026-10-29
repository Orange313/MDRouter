import common as C
import subprocess,os,time,json,hashlib,io,concurrent.futures
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
out=C.ART/'T1_preparation/ocr_001';out.mkdir(parents=True,exist_ok=False)
pk=out/'packages';pk.mkdir();rt=out/'runtime';rt.mkdir()
packages=['tesseract-ocr','libtesseract4','liblept5','tesseract-ocr-eng','tesseract-ocr-osd']
log=subprocess.run(['apt-get','download',*packages],cwd=pk,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);C.prose(out/'package_download.log',log.stdout);assert log.returncode==0
for p in pk.glob('*.deb'):subprocess.run(['dpkg-deb','-x',str(p),str(rt)],check=True)
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(rt/'usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','');env['TESSDATA_PREFIX']=str(rt/'usr/share/tesseract-ocr/4.00/tessdata');env['OMP_THREAD_LIMIT']='1'
exe=rt/'usr/bin/tesseract';version=subprocess.check_output([str(exe),'--version'],env=env,stderr=subprocess.STDOUT,text=True)
C.write(out/'runtime_lock.json',{'time_bj':C.stamp(),'version':version,'package_SHA256':{p.name:C.sha(p) for p in pk.glob('*.deb')},'files_SHA256':{str(p.relative_to(rt)):C.sha(p) for p in rt.rglob('*') if p.is_file()},'engine_args':['-l','eng','--psm','3'],'workers':4,'modifies_shared_venv':False})
cor=C.ROOT/'artifacts/icde_eab_main_006/X3/biomedical_audit_inputs_001/corpus/test-00000-of-00001.parquet'
assert C.sha(cor)=='4559f91bc3a9c4b303e5a0f343e083eeba23f6bc7045ba5cf3602fbe25f95e86'
rows=pq.read_table(cor).to_pylist();pagesdir=out/'png';pagesdir.mkdir();start=time.perf_counter();cpu=time.process_time()
def one(row):
    page=int(row['corpus-id']);png=pagesdir/f'{page}.png';Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB').save(png)
    t=time.perf_counter();a=os.times();p=subprocess.run([str(exe),str(png),'stdout','-l','eng','--psm','3'],env=env,capture_output=True,text=True);b=os.times();assert p.returncode==0
    return {'corpus_id':page,'doc_id':row['doc-id'],'markdown':p.stdout,'OCR_wall_s':time.perf_counter()-t,'OCR_CPU_child_cumulative_delta_s':(b.children_user+b.children_system)-(a.children_user+a.children_system),'image_SHA256':hashlib.sha256(row['image']['bytes']).hexdigest(),'OCR_stderr':p.stderr}
result=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for r in pool.map(one,rows):
        result.append(r)
        if len(result)%100==0:print(json.dumps({'OCR_pages':len(result),'total':len(rows),'time_bj':C.stamp()}),flush=True)
C.write(out/'pages.json',sorted(result,key=lambda r:r['corpus_id']))
times=os.times();C.write(out/'setup_cost.json',{'pages':len(result),'wall_s':time.perf_counter()-start,'parent_CPU_s':time.process_time()-cpu,'child_CPU_s':times.children_user+times.children_system,'GPU_s':0,'OCR_empty_pages':sum(not x['markdown'].strip() for x in result),'note':'per-job child CPU deltas overlap under concurrent workers; aggregate os.times child CPU is authoritative','runtime_lock_SHA256':C.sha(out/'runtime_lock.json'),'pages_SHA256':C.sha(out/'pages.json')})
print(json.dumps({'status':'PASS_OCR_PREPARATION','pages':len(result),'empty':sum(not x['markdown'].strip() for x in result),'wall_s':time.perf_counter()-start,'time_bj':C.stamp()}),flush=True)
