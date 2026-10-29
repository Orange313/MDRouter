import common as C
import pathlib,py_compile
for p in pathlib.Path(__file__).parent.glob('*.py'):py_compile.compile(str(p),doraise=True)
C.write(C.ART/'S1/execution_code_lock_001.json',{'time_bj':C.stamp(),'code_SHA256':{str(p.relative_to(C.ROOT)):C.sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')},'protocol_SHA256':C.sha(C.ART/'S1/streaming_protocol_v2.json'),'note':'operational code locked before any new GPU result; prior inline shell quoting failed before any execution, preserved in launch_failure_001.json'})
C.write(C.ART/'S1/launch_failure_001.json',{'time_bj':C.stamp(),'status':'FAIL_BEFORE_GPU','cause':'PowerShell/ssh inline python quoting SyntaxError; replaced with versioned Python launch script, no scientific change','sealed_reads':0})
print('PASS_CODE_COMPILED_AND_LOCKED',flush=True)
