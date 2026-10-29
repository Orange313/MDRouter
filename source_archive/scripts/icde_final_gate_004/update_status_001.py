import common as C
import json,os,pathlib,sys
tag=sys.argv[1];out=C.ART/'status_snapshots'/tag;out.mkdir(parents=True,exist_ok=False);p=C.ROOT/'manifests/current_experiment_status.json';raw=p.read_bytes();old=json.loads(raw);C.prose(out/'before.json',raw.decode());u=dict(old)
u.update(current_run=C.RUN,latest_run_id=C.RUN,current_phase=tag,updated_at_bj=C.stamp(),last_updated_bj=C.stamp(),latest_reports=str(C.REPORT),new_K1_this_run=False,new_K1_status='NOT_CERTIFIED',new_Eval_status='SEALED_NOT_AUTHORIZED',B_L_unsealing_authorized_this_run=False,B_L_new_raw_reads_this_run=0,old_BL_raw_reads_this_round=0,old_test_CS_query_or_label_reads_this_round=0)
stages={}
for s in ['S0','S1','S2','S3','S4']:
 mp=C.ART/s/'manifest.json'
 if mp.exists():m=json.loads(mp.read_text());stages[s]={'status':m['status'],'SHA256':C.sha(mp),'time_bj':m['time_bj']}
u[C.RUN]={'status':'IN_PROGRESS','phase':tag,'stages':stages,'PharmaCalEval':'SEALED','oldBL_test_CS_reads':0,'no_protocol005_change':True};u['latest_claim_evidence_path']=str(C.REPORT/'claim_to_evidence_matrix.md')
if (C.ART/'manifest.json').exists():mp=C.ART/'manifest.json'
else:mp=C.ART/list(stages)[-1]/'manifest.json'
u.update(manifest=str(mp),current_delivery_manifest=str(mp),manifest_SHA256=C.sha(mp),current_delivery_manifest_SHA256=C.sha(mp))
if tag=='COMPLETED_WITH_LIMITATIONS':u[C.RUN]['status']='COMPLETED_WITH_LIMITATIONS'
C.write(out/'after.json',u);tmp=p.with_name('current_experiment_status.'+C.RUN+'.tmp.json');C.write(tmp,u);assert p.read_bytes()==raw,'Concurrent status update; preserve temp, stop metadata overwrite';os.replace(tmp,p);print(json.dumps({'phase':tag,'time_bj':C.stamp(),'manifest':str(mp)}),flush=True)
