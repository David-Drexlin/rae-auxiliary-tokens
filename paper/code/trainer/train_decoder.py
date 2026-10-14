from __future__ import annotations
import ast, csv, hashlib, importlib, io, json, os, random, platform, sys, time, traceback
from contextlib import nullcontext
from pathlib import Path
from protocol import canonical_sha256, file_sha256, learning_rate, rank_microbatches, read_manifest, seed_for, validate_calibration, validate_runtime_admission
HERE=Path(__file__).resolve().parent
EVALUATION=HERE.parent/'evaluation'

def bind_implementation(binding):
    root = HERE.parent
    paths = sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
    binding['implementation_files'] = {str(p.resolve()): file_sha256(p) for p in paths}
    binding.pop('binding_sha256', None)
    binding['binding_sha256'] = canonical_sha256(binding)
    return binding

def load_rae_parameters(config):
    import copy
    import stage1.rae as module
    from initialization import matched_decoder
    if not getattr(module, '_register_decoder_factory', False):
        original = module.GeneralDecoder
        module.GeneralDecoder = lambda conf, num_patches: matched_decoder(original, conf, num_patches)
        module._register_decoder_factory = True
    params = copy.deepcopy(config['model']['params'])
    fixed = {
        'cross_attn_bidirectional': False,
        'aux_slot_prediction_head': False,
        'aux_slot_prediction_source': 'cls+register',
        'aux_slot_prediction_num_tokens': 5,
        'aux_slot_prediction_layer_index': 2,
        'use_global_cond': False,
    }
    for key, expected in fixed.items():
        if key in params and (type(params[key]) is not type(expected) or params[key] != expected):
            raise ValueError(f'Unsupported model parameter: {key}')
        params.pop(key, None)
    return params

def run_training(config,output,binding,resume=None,stop_after_updates=None,stop_after_epoch=None,verification_callback=None):
    invocation_started=time.perf_counter()
    calibration_gate=validate_calibration(config)
    runtime_gate=validate_runtime_admission(config,binding)
    if config.get('purpose')=='calibration' and stop_after_updates!=128:raise ValueError('Calibration pilots require the frozen --stop-after-updates 128 bound')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from torch import nn
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Dataset
    from PIL import Image
    import numpy as np
    from engine import train_update
    from state import (atomic_json,broadcast_buffers,distributed,frozen_evaluation,load_checkpoint,metadata_group,set_metadata_group,
                       make_ema_after_ddp,restore_rng,save_checkpoint,update_ema)
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local_rank=int(os.environ.get('LOCAL_RANK',0))
    progress=output.parent/'execution_progress'/f'{output.name}_rank{rank}.json';progress.parent.mkdir(parents=True,exist_ok=True)
    def phase(name,**extra):
        atomic_json(progress,{'schema':'scoped-trainer-live-progress-v1','purpose':config.get('purpose'),'condition_id':config['condition_id'],'seed':config['seed'],'job_id':os.environ.get('SLURM_JOB_ID'),'rank':rank,'local_rank':local_rank,'phase':name,'recorded_unix_seconds':time.time(),'optimizer_training':name=='optimizer_training',**extra})
    phase('runtime_initialization')
    if world!=int(config['training']['world_size']):raise ValueError('Runtime world size differs from frozen training.world_size')
    device_kind=config.get('runtime',{}).get('device','cuda')
    if device_kind=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use declared CPU correctness config explicitly')
    device=torch.device('cuda',local_rank) if device_kind=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    binding['runtime']={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'device_type':device.type,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        'world_size':world,'precision':config['training'].get('precision','bf16'),'deterministic':config['training'].get('deterministic',True)}
    if device.type=='cuda':binding['runtime']['transport']=__import__('transport').checkpoint_runtime()
    binding['runtime']['discriminator_reducer_policy']='fixed_declared_buckets_find_unused_true'
    binding['runtime']['preservation_backward_policy']=__import__('preservation_backward').POLICY
    binding.pop('binding_sha256',None);binding['binding_sha256']=canonical_sha256(binding)
    if world>1:
        from datetime import timedelta
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**({'device_id':device} if device.type=='cuda' else {}))
        phase('nccl_process_group_initialized')
        set_metadata_group(dist.new_group(backend='gloo',timeout=timedelta(seconds=180)))
        phase('metadata_process_group_initialized')
        handshake=torch.tensor(rank+1.,device=device)
        dist.all_reduce(handshake)
        if handshake.item()!=world*(world+1)/2:raise RuntimeError('NCCL rank/device tensor handshake differs')
        phase('nccl_tensor_handshake_passed')
    def rank_zero_work(action,label):
        error=None;result=None
        if rank==0:
            try:result=action()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        errors=[error]
        if world>1:dist.broadcast_object_list(errors,src=0,group=metadata_group())
        if errors[0]:raise RuntimeError(label+': '+errors[0])
        return result
    def initialize_output():
        output.mkdir(parents=True,exist_ok=bool(resume))
        if not resume and any(output.iterdir()):raise FileExistsError('New run output must be empty')
        (output/'checkpoints').mkdir(exist_ok=True);(output/'inference').mkdir(exist_ok=True)
        if resume and (output/'RUN_BINDING.json').exists() and json.loads((output/'RUN_BINDING.json').read_text())!=binding:
            raise ValueError('Existing run source/config/data binding differs')
        atomic_json(output/'RUN_BINDING.json',binding)
    rank_zero_work(initialize_output,'Collective output initialization failed')
    phase('output_initialized')
    t=config['training'];seed=int(config['seed']);precision=t.get('precision','bf16')
    torch.manual_seed(seed_for(seed,'model_initialization'));random.seed(seed_for(seed,'python_initialization'))
    np.random.seed(seed_for(seed,'numpy_initialization')%(2**32))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if t.get('deterministic',True):torch.use_deterministic_algorithms(True)
    if device.type=='cpu':torch.set_num_threads(int(config.get('runtime',{}).get('cpu_threads',4)))
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
    if device.type=='cpu' and precision=='fp16':raise ValueError('CPU fp16 is unsupported; choose fp32 or bf16')
    def autocast():return torch.autocast(device.type,dtype=dtype,enabled=precision!='fp32')
    scaler=torch.amp.GradScaler(device.type,enabled=precision=='fp16')
    source=Path(config['source_root']).resolve();sys.path.insert(0,str(source));sys.path.insert(0,str(EVALUATION))
    rae_module=importlib.import_module('stage1.rae')
    if Path(rae_module.__file__).resolve()!=source/'stage1/rae.py':raise RuntimeError('Wrong RAE import source')
    params=load_rae_parameters(config)
    rae=rae_module.RAE(**params).to(device)
    rae.encoder.requires_grad_(False).eval()
    objective_module=importlib.import_module('objectives')
    if Path(objective_module.__file__).resolve()!=EVALUATION/'objectives.py':raise RuntimeError('Wrong objective implementation import')
    objective_cfg={**config['objective'],'source_root':str(source)}


    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        torch.manual_seed(seed_for(seed,'objective_heads'))
        objective=objective_module.build_objective(rae,objective_cfg).to(device)

    class TrainingModule(nn.Module):
        def __init__(self,objective):super().__init__();self.objective=objective
        def train(self,mode=True):
            super().train(mode);self.objective.rae.encoder.eval();return self
        def forward(self,images,include_aux=True):
            value=self.objective(images,include_aux=include_aux)
            return {'reconstruction':value['reconstruction'],'auxiliary_loss':value['auxiliary_loss'],
                    'components':{k:v.detach() if torch.is_tensor(v) else v for k,v in value.get('components',{}).items()}}
    live=TrainingModule(objective).to(device).train()
    disc_module=importlib.import_module('disc')
    disc_cfg=dict(config['discriminator'])
    disc_path=Path(disc_cfg['arch']['dino_ckpt_path']).resolve()
    if str(disc_path) not in {str(Path(p).resolve()) for p in config.get('asset_bindings',{}).values()}:
        raise ValueError('Bind discriminator checkpoint in asset_bindings')
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        torch.manual_seed(seed_for(seed,'discriminator_initialization'))
        discriminator,augment=disc_module.build_discriminator(disc_cfg,device)
    discriminator=discriminator.to(device)
    phase('model_and_discriminator_constructed')
    if world>1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp=DDP(live,device_ids=[device.index] if device.type=='cuda' else None,
                find_unused_parameters=True,broadcast_buffers=False)
        ddp_disc=DDP(discriminator,device_ids=[device.index] if device.type=='cuda' else None,
                     find_unused_parameters=True,broadcast_buffers=False)
    else:ddp=live;ddp_disc=discriminator

    broadcast_buffers(discriminator)
    phase('ddp_initialization_passed')
    ema=make_ema_after_ddp(live)
    trainable=[p for p in live.parameters() if p.requires_grad]
    if not trainable:raise ValueError('No trainable decoder or objective parameters')
    optimizer=torch.optim.AdamW(trainable,lr=float(t.get('lr',2e-4)),betas=tuple(t.get('betas',[.9,.95])),weight_decay=float(t.get('weight_decay',0)))
    disc_optimizer=torch.optim.AdamW([p for p in discriminator.parameters() if p.requires_grad],lr=float(t.get('lr',2e-4)),betas=tuple(t.get('betas',[.9,.95])),weight_decay=0)
    expected_ids={id(p) for p in live.parameters() if p.requires_grad}
    assert expected_ids=={id(p) for group in optimizer.param_groups for p in group['params']}
    base=config.get('base_loss',{})
    lpips=disc_module.LPIPS(use_dropout=True).to(device).eval().requires_grad_(False) if base.get('perceptual_weight',1)>0 else None
    gram=disc_module.GramLoss(layer_ids=(3,8,15),layer_weights=(1.,1.,1.),loss_type='l1').to(device).eval().requires_grad_(False) if base.get('gram_weight',.05)>0 else None
    image_size=int(config['data'].get('image_size',256));resize_shorter=int(config['data'].get('resize_shorter',384))
    rows=read_manifest(config['data']['train_manifest'],True)
    global_batch=int(t['global_batch_size']);accum=int(t.get('grad_accum_steps',1))
    if global_batch%(world*accum):raise ValueError('Global batch incompatible with this world/accumulation')
    steps_per_epoch=len(rows)//global_batch
    if not steps_per_epoch:raise ValueError('Not enough images for one complete optimizer step')
    class ManifestImages(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,item):
            index,epoch=item;row=rows[index]
            raw=Path(row['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Image changed: '+row['stable_id'])
            with Image.open(io.BytesIO(raw)) as opened:
                image=opened.convert('RGB');width,height=image.size
                size=(resize_shorter,int(height*resize_shorter/width)) if width<=height else (int(width*resize_shorter/height),resize_shorter)
                image=image.resize(size,Image.Resampling.BICUBIC)
                crop_rng=random.Random(seed_for(seed,'crop',epoch,row['parent_id'],row['stable_id']))
                left=crop_rng.randrange(size[0]-image_size+1);top=crop_rng.randrange(size[1]-image_size+1)
                array=np.array(image.crop((left,top,left+image_size,top+image_size)),dtype=np.uint8,copy=True)
            return torch.from_numpy(array).permute(2,0,1).float().div_(255),row['stable_id']
    dataset=ManifestImages()
    cursor={'epoch':0,'update_in_epoch':0,'global_update':0,'micro_step':0,'examples_consumed':0,'elapsed_training_seconds':0.0}
    scheduler_state={'completed_updates':0,'steps_per_epoch':steps_per_epoch,'decay_end_epoch':int(t.get('scheduler_decay_end_epoch',t.get('reference_epochs',16)))}
    restored_rng=None
    if resume:
        cursor,scheduler_state,restored_rng=load_checkpoint(resume,live,ema,discriminator,optimizer,disc_optimizer,scaler,binding)
        if scheduler_state['steps_per_epoch']!=steps_per_epoch:raise ValueError('Dataset update count differs from checkpoint')
    torch.manual_seed(seed_for(seed,'training',rank));random.seed(seed_for(seed,'training_python',rank));np.random.seed(seed_for(seed,'training_numpy',rank)%(2**32))
    if restored_rng:
        restore_rng(restored_rng,scaler)
        from state import rng_state,state_value_sha256
        if state_value_sha256(rng_state(scaler))!=state_value_sha256(restored_rng):
            raise ValueError('Exact per-rank RNG restoration failed')
    optimizer.zero_grad(set_to_none=True);disc_optimizer.zero_grad(set_to_none=True)
    segment_start_update=cursor['global_update']
    run_started=time.perf_counter();last_saved=Path(resume).resolve() if resume else None;last_images=None;failure=None
    shared=list(rae.decoder.decoder_layers[int(config.get('diagnostics',{}).get('shared_block_zero_index',0))].parameters())
    diagnostic_every=int(config.get('diagnostics',{}).get('gradient_every_updates',1000))
    checkpoint_epochs=set(int(e) for e in t.get('checkpoint_epochs',[1,2,4,8,12,16,32]))
    save_every=int(t.get('checkpoint_every_updates',0));reference=int(t.get('reference_epochs',16))
    checkpoint_updates=set(map(int,t.get('checkpoint_update_boundaries',[])))
    if checkpoint_updates and config.get('purpose')!='runtime_canary':raise ValueError('Explicit update boundaries are runtime-canary-only')
    def checkpoint(reason):
        nonlocal last_saved
        path=output/'checkpoints'/f'update_{cursor["global_update"]:09d}.pt'
        cursor['elapsed_training_seconds']+=time.perf_counter()-checkpoint.clock
        checkpoint.clock=time.perf_counter()
        if path!=last_saved:
            save_checkpoint(path,live,ema,discriminator,optimizer,disc_optimizer,scaler,binding,dict(cursor),dict(scheduler_state));last_saved=path
        if rank==0:atomic_json(output/'LATEST_CHECKPOINT.json',{'path':str(path),'cursor':cursor,'reason':reason,'binding_sha256':binding['binding_sha256']})
        return path
    checkpoint.clock=time.perf_counter()
    admission_path=config.get('runtime',{}).get('admission_receipt')
    admission=None
    if admission_path:
        admission=json.loads(Path(admission_path).read_text())
    runtime_admitted=bool(runtime_gate['scientific_training_admitted'] and device.type=='cuda')
    def endpoint_receipt(path):
        seal=json.loads(path.with_suffix(path.suffix+'.identity.json').read_text())
        return {'schema_version':'objective-training-endpoint-v1','status':'pass','condition_id':config['condition_id'],'seed':seed,
            'training_completed':True,'epoch':cursor['epoch'],'completed_epoch':cursor['epoch'],
            'checkpoint':str(path),'checkpoint_path':str(path),'checkpoint_sha256':seal['sha256'],
            'source_sha256':binding['source_sha256'],'config_sha256':binding['config_sha256'],
            'implementation_sha256':canonical_sha256(binding['implementation_files']),
            'data_manifest_sha256':binding['assets']['train_manifest']['sha256'],'binding_sha256':binding['binding_sha256'],
            'actual_runtime_admission_receipt':str(Path(admission_path).resolve()) if admission_path else None,
            'actual_runtime_admission_sha256':file_sha256(admission_path) if admission_path else None,
            'actual_runtime_admission_receipt_sha256':file_sha256(admission_path) if admission_path else None,
            'actual_trainer_admitted':runtime_admitted,'scientific_training_admitted':runtime_admitted,
            'purpose':config.get('purpose','scientific'),'calibration_gate':calibration_gate,
            'evaluations':'pending','scientific_result_admitted':False,'training_horizon_remaining':max(0,int(t['epochs'])-cursor['epoch'])}
    if not resume:checkpoint('initial_state')
    phase('ready_for_optimizer_updates',global_update=cursor['global_update'])
    def image_loss(images,reconstruction):
        l1=(reconstruction-images).abs().mean()
        lp=lpips(2*images-1,2*reconstruction-1) if lpips is not None else l1.new_zeros(())
        gr=gram(images,reconstruction) if gram is not None else l1.new_zeros(())
        total=float(base.get('l1_weight',1))*l1+float(base.get('perceptual_weight',1))*lp+float(base.get('gram_weight',.05))*gr
        return total,{'pixel_l1':l1,'lpips':lp,'gram':gr}
    diagnostics_module=importlib.import_module('diagnostics')
    def gradient_metrics(image,aux):
        return diagnostics_module.gradient_diagnostics(image,aux,shared,distributed=False)
    monitor_rows=read_manifest(config['data']['monitor_manifest'],True,unique_parents=False) if config['data'].get('monitor_manifest') else []
    monitored_updates=set()
    crop_tree=ast.parse((source/'utils/train_utils.py').read_text())
    crop_node=next(node for node in crop_tree.body if isinstance(node,ast.FunctionDef) and node.name=='center_crop_arr')
    crop_scope={'Image':Image,'np':np}
    exec(compile(ast.Module(body=[crop_node],type_ignores=[]),str(source/'utils/train_utils.py'),'exec'),crop_scope)
    def monitor():
        if not monitor_rows or cursor['global_update'] in monitored_updates:return
        def write_monitor():
            folder=output/'monitor'/f'update_{cursor["global_update"]:09d}'
            folder.mkdir(parents=True,exist_ok=False)
            started=time.perf_counter();batch=int(config.get('diagnostics',{}).get('monitor_batch_size',8))
            all_original=[];all_reconstruction=[];metric_rows=[]
            with torch.no_grad():
                for start in range(0,len(monitor_rows),batch):
                    selected=monitor_rows[start:start+batch];arrays=[]
                    for row in selected:
                        raw=Path(row['path']).read_bytes()
                        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Monitor image changed: '+row['stable_id'])
                        with Image.open(io.BytesIO(raw)) as image:
                            arrays.append(np.array(crop_scope['center_crop_arr'](image.convert('RGB'),image_size),copy=True))
                    original=np.stack(arrays)
                    images=torch.from_numpy(original).permute(0,3,1,2).to(device).float()/255
                    with autocast():reconstruction=ema(images,include_aux=False)['reconstruction']


                    quantized=torch.round(reconstruction.float().clamp(0,1)*255).to(torch.uint8)
                    metrics=diagnostics_module.image_error_metrics(quantized.float()/255,images,include_dct=True)
                    for index,row in enumerate(selected):
                        metric_rows.append({'stable_id':row['stable_id'],**{key:float(value[index]) if torch.is_tensor(value) else value for key,value in metrics.items()}})
                    all_original.append(original);all_reconstruction.append(quantized.permute(0,2,3,1).cpu().numpy())
            np.savez_compressed(folder/'images.npz',original=np.concatenate(all_original),reconstruction=np.concatenate(all_reconstruction),stable_id=np.array([r['stable_id'] for r in monitor_rows]))
            with (folder/'per_image.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
            atomic_json(folder/'MONITOR_AUDIT.json',{'schema_version':'objective-fixed-monitor-v1','status':'complete',
                'global_update':cursor['global_update'],'epoch_cursor':cursor['epoch'],'count':len(metric_rows),
                'source_manifest_sha256':file_sha256(config['data']['monitor_manifest']),
                'binding_sha256':binding['binding_sha256'],'seconds':time.perf_counter()-started,
                'image_policy':'source_center_crop_then_EMA_decode_round_clamp_uint8',
                'images_sha256':file_sha256(folder/'images.npz'),'metrics_sha256':file_sha256(folder/'per_image.csv'),
                'full_final_evaluation':False,'training_head_predictions':'pending_offline_readout',
                'headfree_parameter_count':sum(p.numel() for p in ema.objective.rae.parameters()),
                'decoder_parameter_count':sum(p.numel() for p in ema.objective.rae.decoder.parameters())})
        rank_zero_work(write_monitor,'Collective monitor output failed')
        monitored_updates.add(cursor['global_update'])
    try:
        stop=(stop_after_epoch is not None and cursor['epoch']>=stop_after_epoch)
        while cursor['epoch']<int(t['epochs']) and not stop:
            epoch=cursor['epoch'];start_update=cursor['update_in_epoch']
            sampler=list(rank_microbatches(len(rows),seed,epoch,global_batch,accum,world,rank,start_update))
            loader=DataLoader(dataset,batch_sampler=sampler,num_workers=int(t.get('workers',t.get('num_workers',8))),pin_memory=device.type=='cuda',
                              generator=torch.Generator().manual_seed(seed_for(seed,'loader',epoch,rank)),persistent_workers=False)
            iterator=iter(loader)
            for update_in_epoch in range(start_update,steps_per_epoch):
                if stop_after_updates is not None and cursor['global_update']>=stop_after_updates:stop=True;break
                update_start=time.perf_counter()
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                lr=learning_rate(cursor['global_update'],steps_per_epoch,t)
                for group in optimizer.param_groups:group['lr']=lr
                for group in disc_optimizer.param_groups:group['lr']=lr
                diagnostic_fn=gradient_metrics if diagnostic_every>0 and cursor['global_update']%diagnostic_every==0 else None
                batch_result=train_update(model=live,ddp_model=ddp,discriminator=discriminator,ddp_discriminator=ddp_disc,
                    optimizer=optimizer,disc_optimizer=disc_optimizer,ema=ema,scaler=scaler,
                    microbatches=(next(iterator) for _ in range(accum)),image_loss_fn=image_loss,augment=augment,
                    device=device,precision=precision,epoch=epoch,base_loss=base,training=t,
                    last_layer=rae.decoder.decoder_pred.weight,diagnostics_fn=diagnostic_fn,teacher=rae.encoder,verification_callback=verification_callback)
                log_values=batch_result['components'];grad_values=batch_result['diagnostics'];last_images=batch_result['last_images']
                cursor.update(global_update=cursor['global_update']+1,update_in_epoch=update_in_epoch+1,micro_step=0,
                              examples_consumed=cursor['examples_consumed']+global_batch)
                scheduler_state['completed_updates']=cursor['global_update']
                if device.type=='cuda':torch.cuda.synchronize()
                record={'recorded_unix_seconds':time.time(),'optimizer_training':True,'global_update':cursor['global_update'],'epoch':epoch,'update_in_epoch':cursor['update_in_epoch'],'lr':lr,
                        'rank':rank,'world_size':world,'global_batch':global_batch,'local_microbatch':global_batch//(world*accum),
                        'contrastive_bank_images':global_batch//accum,
                        'cumulative_gpu_hours':(cursor['elapsed_training_seconds']+time.perf_counter()-checkpoint.clock)*world/3600 if device.type=='cuda' else None,'seconds':time.perf_counter()-update_start,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,**log_values,**grad_values}
                if cursor['global_update']==1 or cursor['global_update']%100==0:
                    phase('optimizer_training',global_update=cursor['global_update'],epoch=epoch,update_in_epoch=cursor['update_in_epoch'],seconds_per_update=record['seconds'],global_batch=global_batch,steps_per_epoch=steps_per_epoch,planned_epochs=int(t['epochs']))

                with (output/f'training_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                monitor_every=int(config.get('diagnostics',{}).get('monitor_every_updates',0))
                if monitor_every>0 and cursor['global_update']%monitor_every==0:monitor()
                if ((save_every>0 and cursor['global_update']%save_every==0) or cursor['global_update'] in checkpoint_updates) and cursor['update_in_epoch']<steps_per_epoch:checkpoint('periodic_update')
            if cursor['update_in_epoch']==steps_per_epoch:
                cursor.update(epoch=epoch+1,update_in_epoch=0)
                if cursor['epoch'] in checkpoint_epochs or cursor['epoch'] in {reference,int(t['epochs'])}:
                    path=checkpoint('epoch_endpoint')
                    monitor()
                    with torch.no_grad(),autocast():
                        wrapped=ema(last_images,include_aux=False)['reconstruction']
                        raw=ema.objective.rae(last_images)
                    identity_error=(wrapped-raw).abs().max().float()
                    if world>1:dist.all_reduce(identity_error,op=dist.ReduceOp.MAX)
                    if not torch.isfinite(identity_error) or float(identity_error)!=0:raise RuntimeError('Head-free inference changed output')
                    export_error=None
                    if rank==0:
                        try:
                            export=output/'inference'/f'epoch_{cursor["epoch"]:03d}.pt'
                            if export.exists():raise FileExistsError(str(export))
                            temp=export.with_suffix('.tmp')
                            torch.save({'schema_version':'objective-headfree-inference-v1','stage_1_params':params,
                                        'state_dict':ema.objective.rae.state_dict(),'binding_sha256':binding['binding_sha256'],
                                        'source_files':binding['source_files'],'source_sha256':binding['source_sha256'],
                                        'config_sha256':binding['config_sha256'],'assets':binding['assets'],
                                        'implementation_files':binding['implementation_files'],
                                        'implementation_sha256':canonical_sha256(binding['implementation_files']),
                                        'model_asset_sha256':{key:binding['assets'][key]['sha256'] for key in ('encoder_weights','encoder_config','encoder_processor','decoder_config','normalization_stats','discriminator_weights','vgg_weights','lpips_weights')},
                                        'head_removal_identity_max_error':float(identity_error)},temp)
                            os.replace(temp,export)
                            endpoint=endpoint_receipt(path)
                            export_sha=file_sha256(export)
                            endpoint.update(inference_path=str(export),inference_sha256=export_sha,inference_checkpoint_path=str(export),inference_checkpoint_sha256=export_sha,head_removal_identity_max_error=float(identity_error))
                            atomic_json(output/f'ENDPOINT_EPOCH_{cursor["epoch"]:03d}.json',endpoint)
                        except Exception as exc:export_error=f'{type(exc).__name__}: {exc}'
                    error_box=[export_error]
                    if world>1:dist.broadcast_object_list(error_box,src=0,group=metadata_group())
                    if error_box[0]:raise RuntimeError('Collective inference export failed: '+error_box[0])
                if stop_after_epoch is not None and cursor['epoch']>=stop_after_epoch:stop=True
        final_checkpoint=checkpoint('complete' if cursor['epoch']>=int(t['epochs']) else 'planned_segment_stop')
        if verification_callback is not None:
            verification_callback('run_complete',{'model':live,'ema':ema,'discriminator':discriminator,'optimizer':optimizer,
                'disc_optimizer':disc_optimizer,'scaler':scaler,'cursor':dict(cursor),'scheduler':dict(scheduler_state),
                'checkpoint':final_checkpoint,'binding':binding,'precision':precision,'teacher':rae.encoder})
        if rank==0 and cursor['epoch']>=int(t['epochs']):
            final_endpoint=json.loads((output/f'ENDPOINT_EPOCH_{cursor["epoch"]:03d}.json').read_text())
            atomic_json(output/'TRAINING_FINAL_AUDIT.json',final_endpoint)
    except Exception:
        failure=traceback.format_exc()
        raise
    finally:
        if rank==0:
            complete=cursor['epoch']>=int(t['epochs']) and failure is None
            total_seconds=time.perf_counter()-invocation_started
            atomic_json(output/f'SEGMENT_COST_{segment_start_update:09d}_{cursor["global_update"]:09d}_{time.time_ns()}.json',
                {'schema_version':'objective-training-segment-cost-v1','start_update':segment_start_update,'end_update':cursor['global_update'],
                 'wall_seconds_including_runtime_setup_monitoring_and_checkpoints':total_seconds,
                 'gpu_hours':total_seconds*world/3600 if device.type=='cuda' else None,
                 'world_size':world,'device_type':device.type,'binding_sha256':binding['binding_sha256']})
            atomic_json(output/'TRAINING_STATE.json',{'schema_version':'objective-training-state-v1','status':'complete' if complete else ('failed' if failure else 'pending'),
                       'cursor':cursor,'last_checkpoint':str(last_saved) if last_saved else None,'binding_sha256':binding['binding_sha256'],
                       'production_validation_claim':False,'scientific_result_admitted':False,'evaluations':'pending',
                       'elapsed_this_segment_seconds':time.perf_counter()-run_started,'failure':failure})
        if distributed():dist.destroy_process_group()
