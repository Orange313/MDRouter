import common as C
import json,pathlib,subprocess
C.start('A0');old=C.ROOT/'artifacts/icde_final_gate_004';checks=[]
def check(p,h,kind):
 a=C.sha(p);checks.append(dict(path=str(p),expected_SHA256=h,actual_SHA256=a,kind=kind,status='PASS' if a==h else 'FAIL'));assert a==h,(p,a,h)
check(old/'manifest.json','1e7cb67f75efa4e8a90c1f0e457930d4416ea80ef31cdf0d0e3b37fc6f71c925','PREDECESSOR')
m=C.load(old/'manifest.json');check(old/'hash_inventory.json',m['inventory_SHA256'],'INVENTORY')
seal=C.load('artifacts/icde_final_gate_004_completion_001/final_integrity_seal.json');special={r['path']:r['final_SHA256'] for r in seal['permitted_final_stdout']}
iv=C.load(old/'hash_inventory.json')
for i,(p,h) in enumerate(iv['files'].items()):
 check(C.ROOT/p,special.get(p,h),'FINAL_SEAL_STDOUT' if p in special else 'PREDECESSOR_OUTPUT')
 if i%1000==0:print(json.dumps({'stage':'A0','checked':i,'total':len(iv['files']),'time_bj':C.stamp()}),flush=True)
for n,h in C.EXPECTED.items():check(C.FROZEN/n,h,'FROZEN005')
check(C.ROOT/'reports/pharma_fixedcorpus_split_audit_001/audit_001/split_manifest.json','a70d0eec64f9611b0371a0f74fcba389f194798fd7b27d5e8e141b71f495440e','SPLIT')
code=C.load(old/'code_final_001/code_commit_record.json');assert code['code_snapshot_commit']==m['code_snapshot_commit'];check(code['bundle_path'],m['code_bundle_SHA256'],'CODE_BUNDLE')
for p,h in code['source_file_SHA256'].items():check(C.ROOT/p,h,'SOURCE_CODE')
status=C.load('manifests/current_experiment_status.json','STATUS_METADATA');C.write(C.A/'status_before.json',status)
for name in ['locked_models.json','domain_policy_config.json','frozen_catalog.json']:
 C.write(C.A/name,C.load(old/'S0'/name))
catalog=C.load(C.A/'frozen_catalog.json');catalog['status']='FROZEN_NO_NEW_MODEL_NO_NEW_CERTIFICATION';catalog['run_id']=C.RUN;catalog['time_bj']=C.stamp()
C.prose(C.R/'frozen_action_catalog.yaml',json.dumps(catalog,ensure_ascii=False,indent=2))
registry=[dict(domain=d,Dev_n=n,candidate_families=f,Fit_n=81 if i==3 else None,prior_exposure='Repeated Dev hyperparameter/engineering exposure; not independent evaluation',generalization='FIXED_CORPUS_QUERY_ONLY',source='icde_final_gate_004/S2/resident_analysis_002',resident_AA='PASS' if i==1 else 'INSUFFICIENT',prior_stream_AA='PASS_2_5_QPS' if i in [0,3] else 'INSUFFICIENT',new_stream_AA='NOT_RUN',IID_effective_n=None) for i,(d,n,f) in enumerate(zip(C.DOM,[40,39,40,45],[24,25,29,44]))]
C.table(C.R/'evidence_registry.csv',registry);C.table(C.R/'integrity_checks.csv',checks)
C.prose(C.R/'data_access_guard.md','# PASS｜访问边界\n\n仅旧三域Dev119、Pharma Fit81/Dev45及无标签固定语料。旧原始B-L/test/CS、Pharma Cal81/Eval73的文本、标签和策略成绩均禁止读取。冻结协议005不变。已观察Dev只支持EXPLORATORY，不将重复请求、随机种子或翻译算作新增独立样本。成本分GPU活动、CPU线程、实际wall与setup摊销；并行资源时间不能相加为E2E，网络与业务净收益未测。\n')
C.write(C.A/'authority_snapshots.json',json.loads((pathlib.Path(__file__).parent/'authority_snapshots.json').read_text(encoding='utf8')))
facts=f'PASS 上轮总manifest、{len(iv["files"])}个封账文件、代码bundle和冻结005/split重新SHA一致；唯一历史stdout追加按既有completion seal核验。固定164条Dev与81条Fit；新Cal/Eval及旧B-L/test/CS读取0。'
C.prose(C.R/'A0_decision.md','# PASS_INPUTS_LOCKED\n\n'+facts+'\n\n进入A1 CPU归因、A2预登记与空闲卡监测、A4论文草稿；不扩大模型。新K1 NOT_CERTIFIED。\n')
C.finish('PASS_INPUTS_LOCKED',facts,'A1/A2/A4并行推进；无可靠A/A禁止Serving优越性主张。')
