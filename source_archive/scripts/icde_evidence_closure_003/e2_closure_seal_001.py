"""Seal quiescent outputs without overwriting the primary E2 inventory."""
import common as C
import pathlib,json
C.start('E2/closure_seal_001');C.STAGE='E2_CLOSURE_SEAL';base=C.ART/'E2';reports=C.REPORT/'E2';old=C.load(reports/'hash_inventory.json','IMMUTABLE_PRIMARY_E2_INVENTORY');C.load(base/'manifest.json','IMMUTABLE_PRIMARY_E2_MANIFEST');mismatches=[]
for rel,h in old['outputs'].items():
 p=C.ROOT/rel;actual=C.sha(p)
 if actual!=h:mismatches.append({'path':rel,'primary_recorded_SHA256':h,'closed_file_SHA256':actual})
allowed={str((base/f).relative_to(C.ROOT)) for f in ['learning_driver_001.log','pipeline_driver_001.log']};assert all(r['path'] in allowed for r in mismatches),'Scientific output changed: stop affected conclusions'
files={str(p.relative_to(C.ROOT)):C.sha(p) for folder in [base,reports] for p in folder.rglob('*') if p.is_file() and C.A not in p.parents and C.R not in p.parents};C.write(C.A/'closed_stage_output_inventory.json',files);C.write(C.R/'live_log_inventory_diagnostic.json',{'time_bj':C.stamp(),'mismatches':mismatches,'reason':'primary process stdout and parent pipeline completion append after finish() hashing; data/model/numerical output files unaffected','repair':'new append-only seal references immutable old manifest and closed actual logs; no result rerun or replacement','scientific_output_mismatch_count':0})
proj=C.load(base/'public_projection_final_001/projection_manifest.json','AUTHORIZED_LABEL_PROJECTION_SCOPE');out=C.load(base/'outcomes_001/outcomes.json','AUTHORIZED_FITDEV_OUTCOMES');C.write(C.R/'pharma_fitdev_manifest.json',{'status':'MATERIALIZED_FIT81_DEV45_ONLY','preflight_manifest':'reports/icde_evidence_closure_003/E2/pharma_fitdev_manifest.json is preserved preflight history','allowed_query_n':126,'qrels_n':636,'Fit_qrels_n':389,'Dev_qrels_n':247,'outcome_n':len(out),'fitdev_qrels_SHA256':proj['fitdev_qrels_SHA256'],'outcomes_SHA256':C.sha(base/'outcomes_001/outcomes.json'),'Cal_Eval_records_read':0,'index_manifest_SHA256':C.sha(C.ROOT/'cache/indexes/pharma_fixedcorpus_index_001/index_manifest.json')})
facts=f'E2封账复核：{len(old["outputs"])}项主输出中，仅{len(mismatches)}条实时stdout/pipeline日志在主manifest计算后追加；全部科学数据/表/索引输出一致。最终静态日志与原manifest通过新seal绑定，旧文件未覆盖。'
decision='工程记录修复，不改变任何数值、模型、路由阈值、协议或标签范围；以新封账清单核验完整E2归档。'
C.prose(C.R/'closure_note.md','# Append-only output sealing\n\n'+facts+'\n\n'+decision+'\n');C.finish('PASS_CLOSED_OUTPUT_SEAL_NO_SCIENTIFIC_CHANGE',facts,decision)
