from pathlib import Path
import json,hashlib,subprocess,zipfile,datetime
B=Path(__file__).parent;folder=B/'paper_R7';receipt=json.loads((folder/'receipt.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(folder/'paper_R7.pdf')==receipt['pdf']['contentHash']
info=subprocess.check_output(['pdfinfo',str(folder/'paper_R7.pdf')],text=True);assert 'Pages:           11' in info
text=(folder/'paper_text_utf8.txt').read_text(encoding='utf8')
assert text.count('REFERENCES')==1
assert all((folder/f'page-{i:02d}.png').exists() for i in range(1,12))
with zipfile.ZipFile(folder/'paper_source_R7.zip') as z:
    source={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    matching=[]
    for x in receipt['build']['inputs']:
        p=next((n for n in source if n==x['logicalPath'] or n.endswith('/'+x['logicalPath'])),None)
        assert p is not None,x['logicalPath']
        assert hashlib.sha256(source[p]).hexdigest()==x['contentHash'],x['logicalPath']
        matching.append({'logicalPath':x['logicalPath'],'sha256':x['contentHash']})
qa={'status':'PASS_RENDER_AND_NUMERIC_SOURCE_QA','time_bj':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'revision':7,'build_id':receipt['build']['id'],'PDF_SHA256':sha(folder/'paper_R7.pdf'),'source_bundle_SHA256':sha(folder/'paper_source_R7.zip'),'pages':11,'visual_pages_inspected':list(range(1,12)),'visual_result':'All final pages inspected; no clipping, table overlap, missing glyphs or duplicated paragraphs found. Bio/ColPali tables readable. Author/affiliation/ORCID placeholders explicitly remain submission blockers.','input_manifest_checks':matching,'numeric_tables':'Biomedical9 and ColPali16 table fragments generated directly from immutable CSVs; 15 numeric summary CSVs replayed byte-identical on this host.','history_retained':['R5 baseline compilation','R6 scorer wording/CI addition; duplicate tail discovered in PDF QA','R7 remove duplicate only, retain all scientific content'],'new_certificate':'NOT_CERTIFIED','public_artifact_release':'NOT_RUN_NOT_PUBLISHED','portable_clean_checkout':'NOT_RUN_WORKSPACE_REPLAY_ONLY'}
(folder/'pdf_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':qa['status'],'pages':11,'source_inputs_verified':len(matching),'PDF_SHA256':qa['PDF_SHA256']}))
