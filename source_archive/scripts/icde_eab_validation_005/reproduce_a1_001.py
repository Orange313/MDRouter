"""CPU raw-measurement replay, no GPU and no sealed inputs. New output directory required."""
import argparse,pathlib,sys,runpy,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
import common as C
C.ROOT=pathlib.Path(a.input_root).resolve();D=pathlib.Path(a.output).resolve();D.mkdir(exist_ok=False);C.ART=D/'artifacts'/C.RUN;C.REPORT=D/'reports'/C.RUN;C.PREV=C.ROOT/'artifacts/icde_evidence_closure_003';C.PR=C.ROOT/'reports/icde_evidence_closure_003'
def finish(status,facts,decision,code=None):
 C.write(D/'replay_receipt.json',{'status':status,'facts':facts,'decision':decision,'time_bj':C.stamp(),'mode':'CPU_REPLAY_OF_LEGAL_ARCHIVED_MEASUREMENTS_NOT_FRESH_HARDWARE','source_code_SHA256':C.sha(pathlib.Path(__file__).parent/'a1_001.py'),'sealed_reads':0})
C.finish=finish
runpy.run_path(str(pathlib.Path(__file__).parent/'a1_001.py'),run_name='__main__')
print(json.dumps({'status':'CPU_REPLAY_COMPLETE','output':str(D)}))
