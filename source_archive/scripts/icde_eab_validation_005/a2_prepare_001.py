import common as C,json,pathlib
C.start('A2');old=C.ROOT/'artifacts/icde_final_gate_004';spec=C.load(old/'S1/streaming_protocol_v2.json');spec.update(run_id=C.RUN,time_bj=C.stamp(),seed=C.SEED,status='PREREGISTERED_V3_BEFORE_NEW_MEASUREMENTS',domains=[C.DOM[0],C.DOM[3]])
spec['version']='v3 repeated validation, same light controls; not tuning for PASS'
spec['post_light_loads']['run_condition']='NOT_RUN_THIS_ROUND; old near-saturation retained'
spec['routing_after_gate']['methods']=['Full-Visual','BM25-R100','BM25','RandomMix_Q_regret','Q_regret','QP_regret'];spec['routing_after_gate']['independent_processes']=3
spec['routing_after_gate']['representative_selection']='all 3-arm router catalog statics plus both frozen learned policies and primary frozen random; BM50 static resident data retained, new stream NOT_RUN to bound budget'
spec['resource_plan']={'max_idle_wait_minutes':45,'allowed_GPUs':[0,1,4],'poll_seconds':10,'idle':'3 consecutive 1s checks: mem<128MiB and utilization=0%; worker repeats before CUDA','external_PID':'abort process and affected domain; no scientific-result retry','first_successful_scheduled_attempt_primary':True,'healthy_run_repeats':0,'pre_CUDA_busy':'wait, no measured rows to select','storage_min_GiB':15}
spec['AA']['random_seed']=C.SEED;spec['AA']['new_AA_required']=True
C.write(C.A/'streaming_protocol_v3.json',spec);C.prose(C.R/'streaming_protocol_v3.yaml',json.dumps(spec,ensure_ascii=False,indent=2))
C.write(C.A/'pre_run_lock.json',{'time_bj':C.stamp(),'protocol_SHA256':C.sha(C.A/'streaming_protocol_v3.json'),'worker_SHA256':C.sha(pathlib.Path(__file__).parent/'stream_worker_v3_001.py'),'models_SHA256':C.sha(C.ART/'A0/locked_models.json'),'code_SHA256':{p.name:C.sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')},'phase':'BEFORE_NEW_GPU_RESULTS'})
print(json.dumps({'status':'PREREGISTERED','time_bj':C.stamp(),'protocol_SHA256':C.sha(C.A/'streaming_protocol_v3.json')}),flush=True)
