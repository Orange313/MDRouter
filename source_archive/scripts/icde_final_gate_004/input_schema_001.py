import common as C
for d in C.DOM[:-1]:
 a=C.load(f'artifacts/icde_main_experiment_001/icde_main_experiment_001_s0_inputs_correctness_001/{d}_dataset.npz');print(d,list(a),flush=True)
features=C.load(C.PREV/'E2/outcomes_001/features.json');print('PharmaFeatures',list(features[0]),flush=True)
