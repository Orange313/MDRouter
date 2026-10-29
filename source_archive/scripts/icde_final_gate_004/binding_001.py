import common as C
import pathlib,json
C.start('S0/asset_binding_001');rows=C.load('reports/icde_decisive_suite_002/R6/binding_004/asset_verification.csv');out=[]
for i,r in enumerate(rows):
 p=pathlib.Path(r['path']);p=p if p.is_absolute() else C.ROOT/p;actual=C.sha(p);assert actual==r['sha256'],(p,actual,r['sha256']);out.append({**r,'actual_SHA256':actual,'reverified_at_bj':C.stamp()})
 if i%1000==0:print(json.dumps({'binding':i,'total':len(rows),'time_bj':C.stamp()}),flush=True)
ip=C.ROOT/'cache/indexes/pharma_fixedcorpus_index_001';m=C.load(ip/'index_manifest.json')
for p,h in m['files'].items():
 actual=C.sha(ip/p);assert actual==h;out.append({'kind':'Pharma_index','path':str(ip/p),'sha256':h,'actual_SHA256':actual,'status':'PASS'})
C.table(C.R/'asset_verification.csv',out);C.write(C.A/'manifest.json',{'status':'PASS','time_bj':C.stamp(),'asset_count':len(out),'verification_csv_SHA256':C.sha(C.R/'asset_verification.csv'),'sealed_content_read':0});print(json.dumps({'status':'PASS_BINDING','assets':len(out),'manifest_sha':C.sha(C.A/'manifest.json')}),flush=True)
