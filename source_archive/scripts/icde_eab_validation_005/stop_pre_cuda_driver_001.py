import pathlib,os,signal,json,common as C
target=str(C.ROOT/'scripts'/C.RUN/'a2_driver_001.py');matches=[]
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   args=(p/'cmdline').read_bytes().split(b'\0')
   if any(a.decode(errors='ignore') in [target,'scripts/'+C.RUN+'/a2_driver_001.py'] for a in args):matches.append(int(p.name))
  except (FileNotFoundError,PermissionError):pass
assert not list((C.ART/'A2').glob('workers/*')),'CUDA or measured phase may have started; do not stop'
assert len(matches)==1,matches
os.kill(matches[0],signal.SIGTERM)
C.write(C.ART/'A2/pre_cuda_driver_amendment.json',{'time_bj':C.stamp(),'old_driver_PID':matches[0],'reason':'Before any GPU work: bind all domain pairs to same physical GPU; initial scheduler could change GPUs between slots','GPU_measurements_before_change':0,'old_attempt_retained':True})
