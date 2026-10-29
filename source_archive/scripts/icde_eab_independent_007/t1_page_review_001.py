import common as C
import pyarrow.parquet as pq
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import io,collections
C.start('T1/page_review_001');pairs=C.load(C.REPORT/'T1/audit_002/perceptual_page_candidates.csv','PERCEPTUAL_SCREEN_CANDIDATES');need=collections.defaultdict(set)
for p in pairs:need[p['old_domain']].add(int(p['old_page']))
inventory=C.load('manifests/primary_datasets_inventory.json');images={};metadata={}
for ds in inventory['datasets']:
    d=ds['repo'].split('/')[-1]
    if d not in need:continue
    for f in ds['files']:
        if not f['path'].startswith('corpus/'):continue
        path=C.ROOT/'data/primary'/d/ds['revision']/f['path'];assert C.sha(path)==f['lfs_sha256']
        for b in pq.ParquetFile(path).iter_batches(batch_size=32,columns=['corpus_id','image','doc_id']):
            for r in b.to_pylist():
                if int(r['corpus_id']) in need[d]:images[d,int(r['corpus_id'])]=Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB');metadata[d,int(r['corpus_id'])]=r['doc_id']
newpng=C.ART/'T1_preparation/ocr_001/png';rendered=[]
for start in range(0,len(pairs),6):
    sheet=Image.new('RGB',(1280,1050),'#cccccc');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(pairs[start:start+6]):
        row=j//2;col=j%2;new=Image.open(newpng/f'{p["Biomedical_page"]}.png').convert('RGB');old=images[p['old_domain'],int(p['old_page'])]
        for k,img in enumerate([new,old]):
            ox=col*640+k*320;oy=row*350;im=ImageOps.contain(img,(310,310));sheet.paste(im,(ox+(320-im.width)//2,oy+25));label=f'{start+j}: Bio {p["Biomedical_page"]}' if k==0 else f'{p["old_domain"].replace("vidore_v3_","")} {p["old_page"]} h={p["pHash_hamming"]}';draw.text((ox+4,oy+5),label,fill='black')
        rendered.append({**p,'old_document':metadata[p['old_domain'],int(p['old_page'])],'review_pair_number':start+j,'contact_sheet':f'contact_{start//6:02d}.png','decision':'PENDING_VISUAL_REVIEW'})
    sheet.save(C.R/f'contact_{start//6:02d}.png')
C.table(C.R/'review_index.csv',rendered);print({'pairs':len(rendered),'contact_sheets':(len(pairs)+5)//6,'path':str(C.R)})
