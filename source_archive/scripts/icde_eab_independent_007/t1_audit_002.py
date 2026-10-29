import common as C
import os,sys,json,collections,hashlib,io,re,time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.fft import dctn
C.start('T1/audit_002');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
old=C.ROOT/'artifacts/icde_eab_main_006/X3/biomedical_audit_inputs_001'
expected={'corpus':'4559f91bc3a9c4b303e5a0f343e083eeba23f6bc7045ba5cf3602fbe25f95e86','docs':'57e29b9d412c46e47b0f2172aaa94aaa9b6e7ed496f14d1f0cc2b650ba464359','queries':'92b9de72e49c72f98e1a2da6b5f9decc2304b1f2ca196d385f0d58ff0174e8a9','qrels':'62934f235e95809cd26fc6e9af982a496012aefad671b9308e008d414c42aa55'}
data={}
for k,h in expected.items():
    p=next((old/k).glob('*.parquet'));assert C.sha(p)==h;C.INPUT[str(p.relative_to(C.ROOT))]={'sha256':h,'role':'PUBLIC_BIOMEDICAL_PROVENANCE_NOT_RETRIEVAL_OUTCOMES'};data[k]=pq.read_table(p).to_pylist()
qs=data['queries'];labels=data['qrels'];corpus=data['corpus'];docs=data['docs']
assert tuple(map(len,[qs,corpus,docs,labels]))==(160,1016,27,515)
pd={int(r['corpus-id']):r['doc-id'] for r in corpus};qdoc=collections.defaultdict(set)
for r in labels:
    if r['score']>0:qdoc[int(r['query-id'])].add(str(pd[int(r['corpus-id'])]))
groups=[{'query_id':int(r['query-id']),'query':r['query'],'query_family':f'english_original_{r["query-id"]}','source_group':'|'.join(sorted(qdoc[int(r['query-id'])])),'source_group_justification':'shared lecture document, descriptive dependence sensitivity not IID count'} for r in qs]
C.table(C.R/'biomedical_query_family_map.csv',groups)
oldq=[]
for d in C.DOM[:3]:oldq += [{'domain':d,'query_id':int(x['query_id']),'query':x['query']} for x in C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json')]
oldq += [{'domain':'Pharma_'+x['role'],'query_id':int(x['query_id']),'query':x['query']} for x in C.load('artifacts/icde_evidence_closure_003/E2/allowed_fitdev_queries.json')]
import torch
from transformers import AutoModel,AutoTokenizer
torch.set_num_threads(4)
mp=Path('/path/to/user/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41') if False else __import__('pathlib').Path('/path/to/user/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);model=AutoModel.from_pretrained(mp,local_files_only=True).eval()
C.write(C.A/'semantic_auditor_lock.json',{'model':'sentence-transformers/all-MiniLM-L6-v2','revision':mp.name,'device':'CPU','threshold':.90,'method':'attention-masked mean pooling L2 normalized; no fitting','files_SHA256':{p.name:C.sha(p) for p in mp.iterdir() if p.is_file()}})
def embed(texts,max_length=256):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(texts),32):
            a=tok(texts[i:i+32],padding=True,truncation=True,max_length=max_length,return_tensors='pt');v=model(**a).last_hidden_state;mask=a['attention_mask'][:,:,None];v=(v*mask).sum(1)/mask.sum(1);out.append(torch.nn.functional.normalize(v,dim=1).numpy())
    return np.concatenate(out)
emb=embed([x['query'] for x in groups+oldq]);sim=emb[:160]@emb.T;near=[]
for i,q in enumerate(groups):
    for j in np.flatnonzero(sim[i]>=.90):
        if j==i or j<160 and j<i:continue
        other=(groups+oldq)[j];near.append({'new_query_id':q['query_id'],'new_query':q['query'],'other_source':'Biomedical' if j<160 else other['domain'],'other_query_id':other['query_id'],'other_query':other['query'],'cosine':float(sim[i,j]),'status':'CANDIDATE_REQUIRES_REVIEW'})
C.table(C.R/'semantic_query_candidates.csv',near,['new_query_id','new_query','other_source','other_query_id','other_query','cosine','status'])
nearest=[]
for i,q in enumerate(groups):
    j=int(np.argmax(sim[i,160:]))+160;other=oldq[j-160];nearest.append({'query_id':q['query_id'],'nearest_old_domain':other['domain'],'nearest_old_query_id':other['query_id'],'cosine':float(sim[i,j])})
C.table(C.R/'query_nearest_old.csv',nearest)
def phash(b):
    a=np.asarray(Image.open(io.BytesIO(b)).convert('L').resize((32,32)),dtype=np.float32);v=dctn(a,norm='ortho')[:8,:8].ravel();return v>np.median(v[1:])
newph=np.array([phash(x['image']['bytes']) for x in corpus]);oldph=[];oldids=[]
inventory=C.load('manifests/primary_datasets_inventory.json');shards=[]
for ds in inventory['datasets']:
    if ds['repo'].split('/')[-1] in C.DOM[:3]:shards += [(ds['repo'].split('/')[-1],C.ROOT/'data/primary'/ds['repo'].split('/')[-1]/ds['revision']/x['path'],x['lfs_sha256']) for x in ds['files'] if x['path'].startswith('corpus/')]
ph=C.ROOT/'data/primary/vidore_v3_pharmaceuticals/262e203d7c59c55947f7042812e0bcc1eee190b1/corpus'
shards += [('Pharma',p,h) for p,h in zip(sorted(ph.glob('*.parquet')),['0f51d221a1def60c6d1436a31507e69be42a6a1da4a9ae45ff6e231751466ac5','827305c58164c9e69581953560b1e76349dbc86a5a06df91d16c92023175e468'])]
import concurrent.futures
image_pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
for name,p,h in shards:
    assert C.sha(p)==h
    for batch in pq.ParquetFile(p).iter_batches(batch_size=32,columns=['corpus_id','image']):
        batch_rows=batch.to_pylist()
        oldph.extend(image_pool.map(phash,[x['image']['bytes'] for x in batch_rows]));oldids.extend((name,int(x['corpus_id'])) for x in batch_rows)
    print(json.dumps({'perceptual_old_pages':len(oldph),'time_bj':C.stamp()}),flush=True)
image_pool.shutdown(wait=True)
oldph=np.array(oldph);pc=[]
for i,v in enumerate(newph):
    dist=np.sum(v!=oldph,axis=1)
    for j in np.flatnonzero(dist<=6):pc.append({'Biomedical_page':int(corpus[i]['corpus-id']),'old_domain':oldids[j][0],'old_page':oldids[j][1],'pHash_hamming':int(dist[j]),'status':'PERCEPTUAL_CANDIDATE_NOT_SEMANTIC_PROOF'})
C.table(C.R/'perceptual_page_candidates.csv',pc,['Biomedical_page','old_domain','old_page','pHash_hamming','status'])
ocr=C.ART/'T1_preparation/ocr_001/pages.json'
deadline=time.monotonic()+1800
while not ocr.exists():
    if time.monotonic()>deadline:raise TimeoutError('OCR not ready for semantic audit')
    time.sleep(5)
pages=C.load(ocr,'UNLABELLED_OCR');texts=[];textids=[]
for idx in ['hr_primary_001','finance_primary_001','industrial_primary_003','pharma_fixedcorpus_index_001']:
    pp=C.load(C.ROOT/'cache/indexes'/idx/'pages.json','UNLABELLED_OLD_CORPUS_TEXT');texts += [x.get('markdown') or '' for x in pp];textids += [(idx,int(x['corpus_id'])) for x in pp]
print(json.dumps({'semantic_page_embeddings':len(texts)+len(pages),'time_bj':C.stamp()}),flush=True)
ne=embed([p['markdown'] for p in pages],128);oe=embed(texts,128);sem=[]
for i,p in enumerate(pages):
    ss=ne[i]@oe.T
    for j in np.flatnonzero(ss>=.90):sem.append({'Biomedical_page':p['corpus_id'],'old_index':textids[j][0],'old_page':textids[j][1],'cosine':float(ss[j]),'status':'OCR_TEXT_SEMANTIC_CANDIDATE_REQUIRES_REVIEW'})
C.table(C.R/'semantic_page_candidates.csv',sem,['Biomedical_page','old_index','old_page','cosine','status'])
summary={'time_bj':C.stamp(),'English_queries':160,'pages':1016,'documents':27,'qrels':515,'qrels_grades':[2],'query_families':160,'source_clusters':27,'exact_old_overlap':'prior audit zero queries, zero exact pages /11609','semantic_query_candidates_ge090':len(near),'perceptual_page_candidates_hamming_le6':len(pc),'OCR_semantic_page_candidates_ge090':len(sem),'visual_semantic_image_encoder_audit':'NOT_RUN; pHash is perceptual and MiniLM audit is OCR-text semantic, not image-semantic proof','pretraining_overlap':'UNKNOWN','MDRouter_prior_benchmark_retrieval_runs':0,'sealed_reads':0}
C.write(C.A/'audit_result.json',summary);C.write(C.A/'audit_inputs.json',C.INPUT);print(json.dumps(summary),flush=True)
