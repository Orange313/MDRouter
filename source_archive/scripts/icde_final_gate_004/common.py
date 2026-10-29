import pathlib,json,csv,hashlib,datetime,sys,subprocess,traceback,io,os
import numpy as np
ROOT=pathlib.Path('/path/to/project')
RUN='icde_final_gate_004'; REPORT=ROOT/'reports'/RUN; ART=ROOT/'artifacts'/RUN
PREV=ROOT/'artifacts/icde_evidence_closure_003'; PR=ROOT/'reports/icde_evidence_closure_003'
SEED=202610271730
DOM=['vidore_v3_hr','vidore_v3_finance_en','vidore_v3_industrial','Pharmaceuticals']
NAMES=['HR','Finance-EN','Industrial','Pharma'];PLANS=['Full-Visual','BM25-R50','BM25-R100','BM25']
FROZEN=ROOT/'reports/prereg_v2_round_004/preregistration_v2_freeze_candidate_005'
EXPECTED={'protocol.json':'ac3284f85a574c3e9443020995ec684e306e06b5c3fa002fa5d95aef7d354d41','asset_lock.json':'116b3a4a486df6049f05221f9cb4856ab5c808605bfa6659d4fca530a9848bbe','freeze_record.json':'f276af09e40cad07c6ad180fc3a8ee8250e6565518b1db865b0f81178a465004'}
INPUT={};A=None;R=None;STAGE=None
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def prose(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:f.write(x)
def table(p,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with pathlib.Path(p).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else clean(v) for k,v in r.items()})
def protected(event,args):
 if event!='open' or not args or not isinstance(args[0],str):return
 p=args[0].replace('\\','/');mode=args[1]
 if isinstance(mode,str) and any(s in mode for s in ['w','a','x']):return
 forbidden=['B-L_pool_queries','final_test_queries','B-L_qrels','B_L_outcomes','/sealed/','pharma_qrels.parquet']
 if any(s in p for s in forbidden) or (('computer_science' in p or '/vidore_v3_cs/' in p) and any(s in p.lower() for s in ['queries','qrels','outcomes'])):raise PermissionError('SEALED INPUT DENIED: '+p)
def start(stage):
 global A,R,STAGE
 STAGE=stage;A=ART/stage;R=REPORT/stage;A.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=False);sys.addaudithook(protected)
 write(A/'stage_start.json',{'stage':stage,'time_bj':stamp(),'seed':SEED,'protected_reads':0})
 ap=A
 def fail(t,v,tb):
  fp=ap/'failure.json'
  if not fp.exists():write(fp,{'status':'FAIL','time_bj':stamp(),'exception':str(v),'traceback':''.join(traceback.format_exception(t,v,tb))})
  traceback.print_exception(t,v,tb)
 sys.excepthook=fail
def load(p,role='AUTHORIZED_DEV_OR_METADATA'):
 p=pathlib.Path(p);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes();rel=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
 INPUT[rel]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'role':role}
 if A is not None:
  with (A/'input_reads.jsonl').open('a') as f:f.write(json.dumps({'time_bj':stamp(),'path':rel,'role':role})+'\n')
 if p.suffix=='.npz':
  with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 if p.suffix=='.json':return json.loads(b)
 if p.suffix=='.csv':return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
 if p.suffix=='.jsonl':return [json.loads(l) for l in b.splitlines()]
 return b.decode('utf8')
def snapshot(tag):
 from archive_code import run
 dest=ART/('code_'+tag);dest.mkdir(exist_ok=False);v=run(ROOT,RUN,dest);return v
def finish(status,facts,decision,code=None):
 code=code or snapshot(STAGE.replace('/','_'))
 inv={'outputs':{str(p.relative_to(ROOT)):sha(p) for folder in [A,R] for p in folder.rglob('*') if p.is_file() and p.name not in ['manifest.json','hash_inventory.json'] and p.suffix!='.log'},'code':code['source_file_SHA256'],'inputs':INPUT,'active_stdout_logs':'outside stage tree; included in final aggregate after process exits'}
 write(R/'hash_inventory.json',inv)
 m={'run_id':RUN,'stage':STAGE,'status':status,'time_bj':stamp(),'code_snapshot_commit':code['code_snapshot_commit'],'base_project_commit':code['shared_project_HEAD_unchanged'],'inventory_SHA256':sha(R/'hash_inventory.json'),'code_bundle_SHA256':code['bundle_SHA256'],'facts':facts,'decision':decision,'frozen005_modified':False,'Cal_Eval_oldBL_test_CS_reads':0,'new_certification':'NOT_CERTIFIED','seed':SEED}
 write(A/'manifest.json',m);write(R/'manifest.json',m)
 text=f'## {RUN} / {STAGE} / {m["time_bj"]}（北京时间）\n\n**结论**：{status}。{facts}\n\n**决策**：{decision}\n\n**阻塞与口径**：Dev条件探索或实际系统测量；重复请求不增加独立query数，区间不构成新风险认证。CPU、GPU、真实wall分别报告；未计网络/完整业务净收益，missing不填0。005未改，Pharma Cal/Eval、旧B-L/test/CS读取0。\n\n**证据**：`{R}/`；manifest `{A}/manifest.json` SHA256 `{sha(A/"manifest.json")}`；代码快照commit `{code["code_snapshot_commit"]}`，共享base `{code["shared_project_HEAD_unchanged"]}`。输入/代码SHA见hash_inventory；依据Orange13 R1、06 R44、10/12。\n'
 prose(R/'orange_entry.md',text);print(json.dumps({**m,'manifest_SHA256':sha(A/'manifest.json')},ensure_ascii=False),flush=True)
def plots():
 sys.path.insert(0,str(ROOT/'cache/report_dependencies/prereg_v2_round_001'));os.environ['MPLCONFIGDIR']=str(A/'matplotlib_cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;return plt
