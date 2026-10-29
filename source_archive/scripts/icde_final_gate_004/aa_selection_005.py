import common as C
import json
def primary(d,b,s):
 valid=[];bad=[]
 for p in sorted((C.ART/'S1/workers').glob(f'{d}_light_b{b}_{s}_*')):
  if (p/'cell_rows.json').exists() and (p/'resource_guard.json').exists() and json.loads((p/'resource_guard.json').read_text())['valid']:valid.append(p)
  else:bad.append(p)
 if not valid:raise RuntimeError(f'INCOMPLETE scheduled pair: {d}/{b}/{s}')
 # First successfully completed scheduled attempt by fixed numeric attempt order, never by performance.
 valid.sort(key=lambda p:int(p.name.rsplit('_',1)[-1]));return valid[0],valid[1:],bad
