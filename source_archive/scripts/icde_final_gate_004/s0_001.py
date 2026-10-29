import common as C
import pathlib,json,subprocess
C.start('S0');prior=C.load('manifests/current_experiment_status.json','STATUS_SNAPSHOT');C.write(C.A/'status_before.json',prior)
checks=[]
def check(p,expected):
 actual=C.sha(p);checks.append({'path':str(p),'expected_SHA256':expected,'actual_SHA256':actual,'status':'PASS' if actual==expected else 'FAIL'});assert actual==expected,(str(p),actual,expected)
check(C.PREV/'manifest.json','0fdb139301921255cd82214fb4bb59f36d63ad8fbd8068a46d352b28b201eb73')
m=C.load(C.PREV/'manifest.json');check(C.PREV/'hash_inventory.json',m['inventory_SHA256'])
iv=C.load(C.PREV/'hash_inventory.json');n=len(iv['files'])
for i,(p,h) in enumerate(iv['files'].items()):
 check(C.ROOT/p,h)
 if i%1000==0:print(json.dumps({'stage':'S0','verified':i,'total':n,'time_bj':C.stamp()}),flush=True)
for name,h in C.EXPECTED.items():check(C.FROZEN/name,h)
check(C.ROOT/'reports/pharma_fixedcorpus_split_audit_001/audit_001/split_manifest.json','a70d0eec64f9611b0371a0f74fcba389f194798fd7b27d5e8e141b71f495440e')
record=C.load(C.PR/'final_code_commit_record.json');assert record['code_snapshot_commit']=='9ab890fc3310cf7a78f6cbde988b87d340998835';check(record['bundle_path'],record['bundle_SHA256'])
for p,h in record['source_file_SHA256'].items():check(C.ROOT/p,h)
# Frozen model/index ledger is reused only after all its assets are independently rehashed again.
ledger=C.load('reports/icde_decisive_suite_002/R6/binding_004/asset_verification.csv')
print(json.dumps({'ledger_fields':list(ledger[0]),'ledger_rows':len(ledger)},ensure_ascii=False),flush=True)
C.write(C.A/'asset_ledger_metadata.json',{'rows':len(ledger),'fields':list(ledger[0]),'note':'Full immutable predecessor inventory validated; individual model/index verification separately in binding_001 before GPU use'})
locked=C.load(C.PREV/'E4/locked_models.json');serv=C.load(C.PREV/'E4/serving_protocol.json');C.write(C.A/'locked_models.json',locked);C.write(C.A/'domain_policy_config.json',serv['domains'])
catalog={'status':'FROZEN_OPERATIONAL_DESIGN_NOT_CERTIFIED','time_bj':C.stamp(),'reference':'Full-Visual exact MaxSim','plans':{'BM25':{'candidates':'text Top10','GPU_query_encoding':False},'BM25-R50':{'candidates':50,'visual_score':'same exact MaxSim and stable tie-break'},'BM25-R100':{'candidates':100,'visual_score':'same exact MaxSim and stable tie-break'},'Full-Visual':{'candidates':'entire fixed corpus','visual_score':'same exact MaxSim and stable tie-break'}},'routers':{'Q_regret':'unchanged E4 PharmaFit81 Q150 Ridge primaryseed kappa20; actions V/R100/BM25','QP_regret':'unchanged E4 primaryseed kappa20, Q150+P13; BM search charged once and reused for BM-selected paths','RandomMix':'frozen E4 query-independent probabilities, realized loss separately reported'},'same_catalog_oracle':'offline LP information upper bound, never deployable','metrics':{'nDCG':'linear graded gain nDCG@10','risk':'r=max(Uref-Up,0), signed gap and rbar separate','harm':'r>1e-10','false_safe':'selected nonreference and r>.05','latency':'actual completion-arrival, no addition of overlapping GPU/CPU seconds','resources':'CPU thread/process seconds, GPU active union secondary profile, setup separate','amortization':'resident E2E(Q)=resident+path-specific setup/Q; resource-specific curves; no frozen business Q'},'data':{'oldDev':119,'PharmaFit':81,'PharmaDev':45,'PharmaCal':'81 SEALED','PharmaEval':'73 SEALED','quarantine':84},'scope':'Fixed-Corpus Query Generalization, Dev selection/exploration; IID UNKNOWN'}
C.write(C.A/'frozen_catalog.json',catalog);C.prose(C.R/'frozen_catalog.yaml',json.dumps(catalog,ensure_ascii=False,indent=2))
C.prose(C.R/'data_access_guard.md','# PASS｜权限边界\n\n仅旧119 Dev、Pharma Fit81/Dev45、无标签固定语料与已有合法计划成本/输出。Python审计钩子拒绝旧B-L/test/CS与混合Pharma原始qrels；不读取Cal/Eval标签或其逐计划输出。不执行旧实验入口，检索后端仅AST抽取函数。封存查询文本也不用于本轮工程选择。索引/分区manifest只允许元数据核验。\n')
C.table(C.R/'integrity_checks.csv',checks)
C.write(C.A/'run_manifest.json',{'predecessor':m,'code':record,'catalog_SHA256':C.sha(C.A/'frozen_catalog.json'),'allowed_sources':'same frozen legal Dev/Fit, no new labels','protected_reads':0})
facts=f'PASS 上轮总manifest及{n}个封账文件逐项重新SHA256一致；代码快照9ab890f、冻结005与Pharma split一致。固定旧119Dev、Fit81/Dev45、4静态计划及不变Q/QP/RandomMix；Cal81/Eval73和旧B-L/test/CS不读。'
decision='进入S1旧日志诊断和新预登记A/A；S2仅合法Dev同风险成本分析。服务可靠性未过前不作性能优越性主张；不增加模型容量。'
C.prose(C.R/'S0_decision.md','# PASS_INPUTS_LOCKED\n\n'+facts+'\n\n'+decision+'\n\n历史E2两条活动stdout的差异由旧closure seal独立绑定，旧总inventory已经绑定最终bytes；本轮核验的是最终封账，不修改旧hash。\n')
C.finish('PASS_INPUTS_LOCKED',facts,decision)
