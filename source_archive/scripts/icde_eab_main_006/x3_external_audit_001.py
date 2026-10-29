import common as C
import pyarrow.parquet as pq
import json,hashlib,re,unicodedata,collections,itertools
import numpy as np
C.A=C.ART/'X3';C.R=C.REPORT/'X3'
out=C.R/'external_audit_001';out.mkdir(exist_ok=False);a=C.A/'biomedical_audit_inputs_001';assert a.exists()
pr=C.load(a/'audit_preregistration.json','PRE_EVALUATION_AUDIT_PROTOCOL');receipt=C.load(a/'acquisition_receipt.json','OFFICIAL_DOWNLOAD_HASHES')
for x in receipt:
    assert x['status']=='PASS';assert C.sha(a/x['file'])==x['sha256']
def read(kind):
    p=next((a/kind).glob('*.parquet'));C.INPUT[str(p.relative_to(C.ROOT))]={'sha256':C.sha(p),'role':'PUBLIC_NEW_DOMAIN_AUDIT_ONLY_NO_RETRIEVAL_EVALUATION'}
    return pq.read_table(p).to_pylist()
qs=read('queries');qrels=read('qrels');docs=read('docs');corpus=read('corpus')
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
querykey='query-id';ck='corpus-id';dk='doc-id'
assert len(qs)==160 and len(docs)==27 and len(corpus)==1016 and len(qrels)==515
querytext={int(x[querykey]):x['query'] for x in qs};dups=[];groups=collections.defaultdict(list)
for q,t in querytext.items():groups[norm(t)].append(q)
for k,v in groups.items():
    if len(v)>1:dups.append({'type':'exact_normalized','query_ids':v})
tokens={q:set(norm(t).split()) for q,t in querytext.items()};near=[]
for u,v in itertools.combinations(querytext,2):
    jac=len(tokens[u]&tokens[v])/max(1,len(tokens[u]|tokens[v]))
    if jac>=.8:near.append({'query1':u,'query2':v,'token_jaccard':jac})
old=[]
for d in C.DOM[:3]:
    old += [(d,int(x['query_id']),x['query']) for x in C.load(f'artifacts/splits/query_generalization_002/{d}/dev_queries.json')]
old += [('Pharma_'+x['role'],int(x['query_id']),x['query']) for x in C.load('artifacts/icde_evidence_closure_003/E2/allowed_fitdev_queries.json')]
cross=[]
for q,t in querytext.items():
    for source,i,ot in old:
        a1,b1=tokens[q],set(norm(ot).split());jac=len(a1&b1)/max(1,len(a1|b1))
        if norm(t)==norm(ot) or jac>=.8:cross.append({'new_query':q,'old_source':source,'old_query':i,'exact':norm(t)==norm(ot),'token_jaccard':jac})
newimages=collections.defaultdict(list)
for x in corpus:
    img=x['image'];b=img['bytes'] if isinstance(img,dict) else img
    newimages[hashlib.sha256(b).hexdigest()].append(int(x[ck]))
pages_by_id={int(x[ck]):x[dk] for x in corpus};positive=[x for x in qrels if int(x.get('score',0))>0]
byq=collections.defaultdict(set)
for x in positive:byq[int(x[querykey])].add(int(x[ck]))
qdocs={q:{pages_by_id[p] for p in ps} for q,ps in byq.items()}
# Source components via positive qrels: descriptive dependence grouping, not IID count.
parent={x[dk]:x[dk] for x in docs}
def find(x):
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
for ds in qdocs.values():
    ds=list(ds)
    for v in ds[1:]:parent[find(v)]=find(ds[0])
components=collections.Counter(find(x) for x in parent)
inventory=C.load('manifests/primary_datasets_inventory.json');shards=[]
for ds in inventory['datasets']:
    if ds['repo'].split('/')[-1] not in C.DOM[:3]:continue
    root=C.ROOT/'data/primary'/ds['repo'].split('/')[-1]/ds['revision']
    shards += [(ds['repo'],root/x['path'],x['lfs_sha256']) for x in ds['files'] if x['path'].startswith('corpus/')]
ph=C.ROOT/'data/primary/vidore_v3_pharmaceuticals/262e203d7c59c55947f7042812e0bcc1eee190b1/corpus'
shards += [('Pharma',p,h) for p,h in zip(sorted(ph.glob('*.parquet')),['0f51d221a1def60c6d1436a31507e69be42a6a1da4a9ae45ff6e231751466ac5','827305c58164c9e69581953560b1e76349dbc86a5a06df91d16c92023175e468'])]
overlap=[];old_pages=0
for name,p,h in shards:
    assert C.sha(p)==h;C.INPUT[str(p.relative_to(C.ROOT))]={'sha256':h,'role':'UNLABELLED_EXISTING_CORPUS_ONLY'}
    for batch in pq.ParquetFile(p).iter_batches(batch_size=32,columns=['corpus_id','image']):
        for x in batch.to_pylist():
            b=x['image']['bytes'];hh=hashlib.sha256(b).hexdigest();old_pages+=1
            if hh in newimages:overlap.append({'old_domain':name,'old_page':int(x['corpus_id']),'new_pages':newimages[hh],'image_SHA256':hh})
    print(json.dumps({'old_unlabelled_corpus_audited':old_pages,'time_bj':C.stamp()}),flush=True)
C.table(out/'query_exact_duplicates.csv',dups,['type','query_ids']);C.table(out/'query_near_candidates.csv',near,['query1','query2','token_jaccard']);C.table(out/'query_overlap_with_allowed_fitdev.csv',cross,['new_query','old_source','old_query','exact','token_jaccard']);C.table(out/'page_exact_overlap.csv',overlap,['old_domain','old_page','new_pages','image_SHA256'])
summary={'status':'PASS_OFFICIAL_INPUT_AUDIT_INSUFFICIENT_MODEL_TRAINING_OVERLAP','repo':pr['repo'],'revision':pr['revision'],'license':'CC-BY-NC-SA-4.0','actual_pages':len(corpus),'actual_documents':len(docs),'actual_English_queries':len(qs),'qrels_rows':len(qrels),'qrels_score_values':dict(collections.Counter(str(x['score']) for x in qrels)),'positive_qrels_rows':len(positive),'queries_with_positive_qrels':len(byq),'multi_relevant_queries':sum(len(x)>1 for x in byq.values()),'multi_source_queries':sum(len(x)>1 for x in qdocs.values()),'source_components':len(components),'exact_query_duplicate_groups':len(dups),'near_query_candidates_jaccard_ge08':len(near),'overlap_with_existing_allowed_245_queries':len(cross),'old_pages_exact_image_audited':old_pages,'exact_cross_corpus_page_overlap':len(overlap),'model_training_overlap':'UNKNOWN_NOT_CLEARED','semantic_image_near_duplicate_audit':'NOT_RUN','query_generation':'ViDoRe V2 synthetic generation plus human review per official publication/card; not IID production query labels','retrieval_evaluation_reads_or_runs':0,'decision':'NO_NEW_BENCHMARK_EVALUATION_UNTIL_INDEPENDENCE_SCOPE_CLEAR_AND_EXECUTABLE_PROTOCOL_LOCKED'}
C.write(out/'audit_summary.json',summary)
C.prose(C.R/'external_domain_readiness.md','# INSUFFICIENT｜外部英文域审计完成，评价未启动\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n统计口径：160条原始英文问题不是多语言160×N；exact/token-Jaccard是保守候选审计，不能证明语义独立。来源组件只作敏感性。官方仓库名下test指公开benchmark发布split，本轮没有读取项目封存test。许可仅用于本学术审计，不向公开artifact转发原始页面；许可与MIT课程底层资料均保留署名与获取链接。四官方LFS文件hash一致，license文本单独hash。已登记只读审计协议后取得标签schema/来源，不执行检索成绩或调参。模型训练集完整页面/URL没有完成交叉验证，公开基座预训练污染也无法排除，因此暂不称独立或训练未见benchmark。下一步应补训练清单/页面URL或权威分割证据，预先锁定一次性描述评价范围；不能依据收益换域。\n')
print(json.dumps(summary),flush=True)
