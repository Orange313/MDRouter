import common as C
import numpy as np
import json
C.start('A2_reporting_numbers_002')
requests=C.load(C.REPORT/'A2/new_valid_requests.csv')
cells=C.load(C.REPORT/'A2/aa_cells_v3.csv')
energy=C.load(C.REPORT/'A2/policy_analysis_001/whole_board_energy_by_cell.csv')
output=[]
for domain in [C.DOM[0],C.DOM[3]]:
 for qps in [2.,5.]:
  selected=[r for r in requests if r['domain']==domain and float(r['QPS_offered'])==qps]
  def values(key):return np.array([float(r[key]) for r in selected])
  c=[r for r in cells if r['domain']==domain and float(r['QPS_offered'])==qps]
  e=[float(r['board_energy_J_per_arrival']) for r in energy if r['domain']==domain and float(r['QPS'])==qps and r['board_energy_J_per_arrival']]
  output.append({
   'domain':domain,'QPS':qps,'requests':len(selected),
   'mean_service_ms':1000*values('service_wall_s').mean(),
   'mean_arrival_ms':1000*values('arrival_to_completion_s').mean(),
   'P50_ms':1000*np.quantile(values('arrival_to_completion_s'),.5),
   'P95_ms':1000*np.quantile(values('arrival_to_completion_s'),.95),
   'mean_CPU_thread_ms':1000*values('service_thread_CPU_s').mean(),
   'mean_completed_QPS':np.mean([float(r['completed_QPS']) for r in c]),
   'mean_goodput_QPS_le200ms':np.mean([float(r['goodput_QPS_le200ms']) for r in c]),
   'board_energy_J_per_arrival':np.mean(e) if e else None,
   'scope':'newFullVAA only; pooled request quantiles descriptive, board energy includesidle'
  })
C.table(C.R/'AA_actual_cost_summary.csv',output)
C.write(C.A/'numbers.json',output)
print(json.dumps(C.clean(output)),flush=True)
