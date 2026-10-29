import pathlib,json,csv,hashlib,datetime,sys,subprocess,traceback,io
import numpy as np
ROOT=pathlib.Path('/path/to/project')
RUN='icde_evidence_closure_003'; REPORT=ROOT/'reports'/RUN; ART=ROOT/'artifacts'/RUN
OLD=ROOT/'reports/icde_main_experiment_001'; SEED=202610271130
DOM=['vidore_v3_hr','vidore_v3_finance_en','vidore_v3_industrial']; NAMES=['HR','Finance-EN','Industrial']
RANK=dict(zip(DOM,['artifacts/phase1/hr_pilot_004/rankings.jsonl','artifacts/phase1/finance_pilot_001/rankings.jsonl','artifacts/phase1/industrial_pilot_003/rankings.jsonl']))
DEV={d:f'artifacts/{"cross_domain_k1_001" if d==DOM[1] else "cross_domain_k1_resume_001"}/development_analysis_001/{d}/counterfactual_complete.json' for d in DOM}
FROZEN=ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
EXPECTED={'protocol.json':'ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41','asset_lock.json':'116b3a4a486df6049f05221f9cb4856ab5c808605bfa6659d4fca530a9848bbe','freeze_record.json':'f276af09e40cad07c6ad180fc3a8ee8250e6565518b1db865b0f81178a465004'}
INPUT={}; STAGE=None; A=None; R=None
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def prose(p,x):
 with pathlib.Path(p).open('x',encoding='utf8') as f:f.write(x)
def table(p,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with pathlib.Path(p).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else clean(v) for k,v in r.items()})
def load(p,role='DEV_OR_METADATA'):
 p=pathlib.Path(p)
 if not p.is_absolute():p=ROOT/p
 b=p.read_bytes();INPUT[str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'role':role}
 with (A/'input_reads.jsonl').open('a') as f:f.write(json.dumps({'time_bj':stamp(),'path':str(p.relative_to(ROOT)),'role':role})+'\n')
 if p.suffix=='.npz':
  with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 if p.suffix=='.json':return json.loads(b)
 if p.suffix=='.csv':return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
 if p.suffix=='.jsonl':return [json.loads(l) for l in b.splitlines()]
 return b.decode('utf8')
def start(stage):
 global STAGE,A,R
 STAGE=stage;A=ART/stage;R=REPORT/stage;A.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=False)
 def guard(event,args):
  if event!='open' or not args or not isinstance(args[0],str):return
  p=args[0].replace('\\','/');mode=args[1]
  if isinstance(mode,str) and any(v in mode for v in ['w','a','x']):return
  if any(s in p for s in ['B-L_pool_queries','final_test_queries','B-L_qrels','B_L_outcomes','/sealed/','pharma_qrels.parquet']):raise PermissionError('SEALED INPUT: '+p)
  if ('computer_science' in p or '/vidore_v3_cs/' in p) and any(s in p.lower() for s in ['queries','qrels','outcomes']):raise PermissionError('SEALED CS: '+p)
 sys.addaudithook(guard)
 write(A/'stage_start.json',{'stage':stage,'time_bj':stamp(),'seed':SEED,'sealed_reads':0})
 def failure(t,v,tb):
  write(A/'failure.json',{'status':'FAIL','time_bj':stamp(),'exception':str(v),'traceback':''.join(traceback.format_exception(t,v,tb))});traceback.print_exception(t,v,tb)
 sys.excepthook=failure
def finish(status,facts,decision):
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip() or 'NO_GIT_COMMIT'
 files={str(p.relative_to(ROOT)):sha(p) for directory in [A,R] for p in directory.rglob('*') if p.is_file() and p.name not in ['manifest.json','hash_inventory.json']}
 codes={str(p.relative_to(ROOT)):sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')}
 inv={'outputs':files,'code':codes,'inputs':INPUT};write(R/'hash_inventory.json',inv)
 m={'run_id':RUN,'stage':STAGE,'status':status,'time_bj':stamp(),'git_commit':commit,'code_SHA256':codes,'input_SHA256':INPUT,'inventory_SHA256':sha(R/'hash_inventory.json'),'facts':facts,'decision':decision,'old_005_modified':False,'new_old_BL_raw_reads':0,'old_test_CS_label_query_reads':0,'new_certification':False,'seed':SEED}
 write(A/'manifest.json',m);write(R/'manifest.json',m)
 entry=f'## {RUN} / {STAGE} / {stamp()}（北京时间）\n\n'
 entry+='**结论**：'+status+'。'+facts+'\n\n**决策**：'+decision+'\n\n**阻塞与口径**：本阶段 Dev 重放/诊断不构成独立认证；GPU 活动秒与 CPU 时间分列，missing 不填0。005 不改，旧原始B-L/test/CS查询标签读取均0。\n\n'
 entry+=f'**证据**：`{R}/`；manifest `{A}/manifest.json` SHA256 `{sha(A/"manifest.json")}`；git `{commit}`，逐输入/代码/分区SHA见 hash_inventory.json。详细表图和失败记录保存在本阶段目录。依据 Orange11 R1、10 R1。\n'
 prose(R/'orange_entry.md',entry)
 print(json.dumps({'stage':STAGE,'status':status,'facts':facts,'decision':decision,'manifest_sha256':sha(A/'manifest.json'),'time_bj':stamp()},ensure_ascii=False),flush=True)
