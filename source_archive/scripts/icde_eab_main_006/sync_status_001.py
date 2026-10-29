import common as C
import json,os,sys
dest=C.ROOT/'artifacts/icde_eab_main_006_completion_001';dest.mkdir(exist_ok=False)
p=C.ROOT/'manifests/current_experiment_status.json';s=json.loads(p.read_text());C.write(dest/'status_before.json',s);root=json.loads((C.ART/'manifest.json').read_text());msha=C.sha(C.ART/'manifest.json')
s['current_run_id']=C.RUN;s['current_stage']='CLOSED_CPU_COMPLETE_GPU_BLOCKED';s['updated_time_bj']=C.stamp();s['current_run_status']=root['status'];s['current_run_manifest']=str(C.ART/'manifest.json');s['current_run_manifest_SHA256']=msha;s['current_run_code_commit']=root['code_snapshot_commit'];s['current_orange06_revision']=int(sys.argv[1]);s['current_experiment_summary']={'Dev_queries':164,'new_GPU_quality_queries':0,'GPU':'BLOCKED_AFTER_45MIN_134_POLLS','new_K1':'NOT_CERTIFIED','Pharma_Cal_Eval':'SEALED','old_BL_test_CS':'SEALED','new_forbidden_label_reads':0,'paper_revision':4,'paper_pages':10,'independent_new_benchmark_evaluation':'NOT_RUN','EAB':'CONDITIONAL_NOT_READY'}
C.write(dest/'status_after.json',s);tmp=dest/'status_atomic_write.json';C.write(tmp,s);os.replace(tmp,p)
C.write(dest/'status_sync_receipt.json',{'time_bj':C.stamp(),'root_manifest_SHA256':msha,'current_status_SHA256':C.sha(p),'Orange06_revision':int(sys.argv[1]),'preserved_historical_fields':True})
print(json.dumps({'status':'STATUS_SYNCED','manifest_SHA256':msha,'current_status_SHA256':C.sha(p)}),flush=True)
