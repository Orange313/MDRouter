import common as C
import collections
C.A=C.ART/'T1';C.R=C.REPORT/'T1';rows=C.load(C.R/'audit_002/semantic_page_candidates.csv','SEMANTIC_SCREEN');new={int(x['corpus_id']):x['markdown'] for x in C.load(C.ART/'T1_preparation/ocr_001/pages.json','UNLABELLED_OCR')};old={}
for name in ['hr_primary_001','finance_primary_001','industrial_primary_003','pharma_fixedcorpus_index_001']:
    for x in C.load(C.ROOT/'cache/indexes'/name/'pages.json','UNLABELLED_OLD_CORPUS_TEXT'):old[name,int(x['corpus_id'])]=x.get('markdown') or ''
counts=collections.Counter();unresolved=[]
for r in rows:
    a=new[int(r['Biomedical_page'])];b=old[r['old_index'],int(r['old_page'])]
    if not a.strip() or not b.strip():status='NONINFORMATIVE_EMPTY_TEXT_EMBEDDING_COLLISION'
    else:status='UNRESOLVED_NONEMPTY_TEXT_NEAR_DUPLICATE';unresolved.append(r)
    counts[status]+=1;r.update({'review_status':status,'new_OCR_chars':len(a.strip()),'old_text_chars':len(b.strip()),'basis':'MiniLM on empty strings has a fixed embedding; not evidence of matching page semantics. Pages/queries remain included; image exact overlap0 and perceptual-screen review separate.','review_time_bj':C.stamp()})
C.table(C.R/'semantic_page_adjudication.csv',rows);C.write(C.A/'semantic_adjudication_receipt.json',{'time_bj':C.stamp(),'counts':dict(counts),'unresolved':len(unresolved),'candidate_pairs':len(rows),'distinct_new_pages':len({x['Biomedical_page'] for x in rows}),'scope':'screening-input degeneracy, not removal of unfavorable evaluation data'})
print({'counts':dict(counts),'unresolved':len(unresolved)})
