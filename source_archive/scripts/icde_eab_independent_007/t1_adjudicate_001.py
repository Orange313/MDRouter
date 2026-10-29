import common as C
C.A=C.ART/'T1';C.R=C.REPORT/'T1'
rows=C.load(C.R/'page_review_002/review_index.csv','CODEX_VISUAL_REVIEW_CONTACT_SHEETS');assert len(rows)==62
def reason(i):
    if i<=21:return 'Sparse copyright-removed biomedical slide versus distinct HR/finance/manual text or mechanical figure; different visible subject, document and content.'
    if i<=30:return 'Biomedical fibroblast migration chart and caption versus securities/legal/industrial text; chart and text are visibly different.'
    if i==31:return 'Scaffold cell-migration explanatory paragraph versus industrial O-ring figure and manual text.'
    if i<=33:return 'MIT OpenCourseWare source/lecture title versus financial payment instructions.'
    if i==34:return 'Copyright-removed textile vascular prosthesis note versus aircraft laminate construction photograph.'
    if i<=38:return 'Premature-baby fleece/photo-removed slide versus securities certificates and legal signatures.'
    if i==39:return 'Patient skin graft/photo-removed slide versus industrial jumper-wire procedural table.'
    if i==40:return 'Wound contraction graph versus industrial contract paragraph.'
    if i==41:return 'Organ-regeneration scaffold question/bullets versus dense methodological pharmaceutical chapter; no repeated visible figure or paragraph.'
    if i==42:return 'Biologically active materials title/bullets versus HR mutual-learning annex.'
    if i<=49:return 'Collagen in-vitro/in-vivo degradation plot versus financial table or repetitive securities legal text.'
    if i==50:return 'Collagen melting temperature plot versus finance contractual prose.'
    if i<=53:return 'Scaffold pore structure/copyright-removed diagram slide versus finance law text or industrial tools specification.'
    if i==54:return 'Blue MIT tissue-types title slide versus industrial fastener table.'
    if i==55:return 'Embryonic germ-disc list versus pharmaceutical antimicrobial prose, distinct content and layout.'
    if i==56:return 'Fibroblast citation/copyright placeholder versus industrial tool handling text and workshop photograph.'
    if i==57:return 'Blue orthopaedic metals four-slide page versus pharmaceutical immunology paragraph page.'
    if i<=60:return 'Blue polyethylene radiation/crosslinking slide montage versus finance definitions, pharmaceutical public-health prose or acknowledgments.'
    return 'Polyethylene crystallography slide montage versus pharmaceutical fairness/information-use prose.'
for r in rows:
    i=int(r['review_pair_number']);r.update({'decision':'NOT_DUPLICATE_EXPLAINED_PERCEPTUAL_HASH_COLLISION','reviewer':'Codex visual inspection, not human annotation','review_time_bj':C.stamp(),'basis':reason(i),'limitation':'Screened62 candidates only; does not prove no semantic overlap or pretraining contamination'})
C.table(C.R/'page_candidate_adjudication.csv',rows)
C.write(C.A/'visual_review_receipt.json',{'time_bj':C.stamp(),'inspected_contact_sheets':[f'contact_{i:02d}.png' for i in range(11)],'inspected_pairs':62,'unexplained_candidates':0,'evidence_SHA256':{p.name:C.sha(p) for p in (C.R/'page_review_002').glob('contact_*.png')},'judgment':'AI-assisted visual source/content review; no human certification','new_finding':'Many released Biomedical slides explicitly remove photographs/diagrams for copyright. Keep all pages/queries; report this content limitation.'})
print({'pairs':len(rows),'unexplained':0,'status':'PASS_SCREENED_CANDIDATE_REVIEW_ONLY'})
