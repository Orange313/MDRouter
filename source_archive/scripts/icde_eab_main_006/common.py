"""Versioned run utilities. Exclusive outputs; explicit guarded inputs."""
from pathlib import Path
import csv, datetime, hashlib, io, json, os, subprocess, sys, traceback
import numpy as np
ROOT=Path('/path/to/project')
RUN='icde_eab_main_006'
ART=ROOT/'artifacts'/RUN
REPORT=ROOT/'reports'/RUN
SEED=202610272130
DOM=['vidore_v3_hr','vidore_v3_finance_en','vidore_v3_industrial','Pharmaceuticals']
NAMES=['HR','Finance-EN','Industrial','Pharma']
PLANS=['Full-Visual','BM25-R50','BM25-R100','BM25']
A=R=None
INPUT={}
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f: json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def prose(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(s)
def table(p,rows,fields=None):
    rows=list(rows); fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else clean(v) for k,v in r.items()})
def denied(p):
    s=str(p).replace('\\','/').lower()
    return any(x in s for x in ['/sealed/','b-l_qrels','b-l_pool_queries','b_l_outcomes','final_test_queries','pharma_qrels.parquet','/prereg_v2_round_005/']) or (any(x in s for x in ['computer_science','/vidore_v3_cs/']) and any(x in s for x in ['queries','qrels','outcomes'])) or (any(x in s for x in ['pharma','pharmaceutical']) and any(x in s for x in ['eval_qrels','cal_qrels','eval_outcomes','cal_outcomes','eval_queries','cal_queries']))
def protected(event,args):
    if event!='open' or not args or not isinstance(args[0],str):return
    mode=args[1] if len(args)>1 else ''
    if isinstance(mode,str) and any(x in mode for x in ['w','a','x']):return
    if denied(args[0]):raise PermissionError('SEALED READ REFUSED BEFORE OPEN')
def start(stage):
    global A,R
    A=ART/stage;R=REPORT/stage
    A.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=False)
    sys.addaudithook(protected)
    write(A/'stage_start.json',{'time_bj':stamp(),'seed':SEED,'stage':stage,'scope':'Dev only; no certification'})
    def fail(t,v,tb):
        if not (A/'failure.json').exists():write(A/'failure.json',{'time_bj':stamp(),'exception':str(v),'traceback':''.join(traceback.format_exception(t,v,tb))})
        traceback.print_exception(t,v,tb)
    sys.excepthook=fail
def load(p,role='AUTHORIZED_DEV_OR_METADATA'):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    if denied(p):raise PermissionError('SEALED READ REFUSED BEFORE OPEN')
    b=p.read_bytes();rel=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    INPUT[rel]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'role':role}
    if A is not None:
        with (A/'input_reads.jsonl').open('a') as f:f.write(json.dumps({'time_bj':stamp(),'path':rel,'role':role,'sha256':INPUT[rel]['sha256']})+'\n')
    if p.suffix=='.json':return json.loads(b)
    if p.suffix=='.csv':return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
    if p.suffix=='.jsonl':return [json.loads(x) for x in b.splitlines()]
    if p.suffix=='.npz':
        with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    return b.decode('utf8')
def snapshot(stage):
    from archive_code import run
    return run(ROOT,RUN,ART/('code_'+stage))
def finish(stage,status,facts,decision):
    code=snapshot(stage)
    entry=f'## {RUN} / {stage} / {stamp()}（北京时间）\n\n**结论**：{status}。{facts}\n\n**决策**：{decision}\n\n**阻塞与口径**：四域164 Dev/122候选族为 DEV_EXPLORATORY；重复计时不增加独立n。新K1 NOT_CERTIFIED，Pharma Cal81/Eval73及旧B-L/test/CS读取0。CPU/GPU/wall分列，Probe实际计费；不声称业务净收益或未经AA核验的Serving优越性。\n\n**证据**：`{R}`；code commit `{code["code_snapshot_commit"]}`；manifest `artifacts/{RUN}/{stage}/manifest.json`，SHA见随本条提交的回执；依据Orange16 R2、15 R2、14 R2、06 R54。\n'
    prose(R/'orange_entry_base.md',entry)
    inv={'outputs':{str(p.relative_to(ROOT)):sha(p) for folder in [A,R] for p in folder.rglob('*') if p.is_file() and p.name not in ['hash_inventory.json','manifest.json']},'inputs':INPUT,'code':code['source_file_SHA256']}
    write(R/'hash_inventory.json',inv)
    m={'run_id':RUN,'stage':stage,'status':status,'time_bj':stamp(),'seed':SEED,'facts':facts,'decision':decision,'code_snapshot_commit':code['code_snapshot_commit'],'base_project_commit':code['base_project_commit'],'code_bundle_SHA256':code['bundle_SHA256'],'inventory_SHA256':sha(R/'hash_inventory.json'),'sealed_reads':0,'new_certification':'NOT_CERTIFIED'}
    write(A/'manifest.json',m);write(R/'manifest.json',m)
    print(json.dumps({**m,'manifest_SHA256':sha(A/'manifest.json')},ensure_ascii=False),flush=True)
