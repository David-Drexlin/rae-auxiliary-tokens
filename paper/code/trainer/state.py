from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import torch
from torch import nn
import torch.distributed as dist
from protocol import file_sha256


_METADATA_GROUP=None
def metadata_group():return _METADATA_GROUP
def set_metadata_group(group):
    global _METADATA_GROUP
    _METADATA_GROUP=group

def distributed():return dist.is_available() and dist.is_initialized()
def rank():return dist.get_rank() if distributed() else 0

def broadcast_buffers(module):
    if distributed():
        for value in module.buffers():dist.broadcast(value,src=0)

@contextmanager
def frozen_evaluation(module):
    flags=[p.requires_grad for p in module.parameters()]
    modes={m:m.training for m in module.modules()}
    try:
        module.eval()
        module.requires_grad_(False)
        yield module
    finally:
        for p,flag in zip(module.parameters(),flags):p.requires_grad_(flag)
        for m,mode in modes.items():m.training=mode


def make_ema_after_ddp(module):
    ema=copy.deepcopy(module).eval().requires_grad_(False)
    return ema

@torch.no_grad()
def update_ema(ema,module,decay):
    source=dict(module.named_parameters())
    for name,value in ema.named_parameters():
        live=source[name]
        if live.requires_grad:value.mul_(decay).add_(live.detach(),alpha=1-decay)
        else:value.copy_(live.detach())
    source_buffers=dict(module.named_buffers())
    for name,value in ema.named_buffers():value.copy_(source_buffers[name])


def rng_state(scaler=None):
    state={'python':random.getstate(),'torch_cpu':torch.get_rng_state(),'scaler':scaler.state_dict() if scaler else None}
    if torch.cuda.is_available():state['torch_cuda']=torch.cuda.get_rng_state(torch.cuda.current_device())
    try:
        import numpy as np
        state['numpy']=np.random.get_state()
    except ImportError:pass
    return state

def restore_rng(state,scaler=None):
    random.setstate(state['python']);torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state:torch.cuda.set_rng_state(state['torch_cuda'],torch.cuda.current_device())
    if state.get('scaler') is not None:
        if scaler is None:raise ValueError('Checkpoint requires its mixed-precision scaler')
        scaler.load_state_dict(state['scaler'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)


def _collective_preflight(error):
    errors=[None]*(dist.get_world_size() if distributed() else 1)
    if distributed():dist.all_gather_object(errors,error,group=metadata_group())
    else:errors[0]=error
    if any(errors):raise ValueError('Collective checkpoint preflight rejected: '+json.dumps({str(i):e for i,e in enumerate(errors) if e}))


def state_value_sha256(value):

    digest=hashlib.sha256()
    def frame(kind,data):
        encoded=kind.encode()+b':' + str(len(data)).encode()+b':'
        digest.update(encoded);digest.update(data)
    def visit(item):
        if torch.is_tensor(item):
            tensor=item.detach().contiguous().cpu()
            frame('tensor_schema',json.dumps([str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
            frame('tensor_bytes',tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            frame('mapping',str(len(item)).encode())
            for key in sorted(item,key=lambda key:(type(key).__name__,repr(key))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            frame(type(item).__name__,str(len(item)).encode())
            for entry in item:visit(entry)
        elif item is None or isinstance(item,(str,int,float,bool)):
            frame(type(item).__name__,json.dumps(item,allow_nan=False,separators=(',',':')).encode())
        else:

            if hasattr(item,'dtype') and hasattr(item,'shape') and hasattr(item,'tobytes'):
                frame('array_schema',json.dumps([str(item.dtype),list(item.shape)],separators=(',',':')).encode());frame('array_bytes',item.tobytes())
            else:raise TypeError('Unsupported state hash value: '+type(item).__name__)
    visit(value)
    return digest.hexdigest()


def initial_state_sha256(payload):

    value={key:payload[key] for key in ('model','ema','discriminator','optimizer','discriminator_optimizer')}
    value['rank_scalers']=[state.get('scaler') for state in payload['rank_states']]
    return state_value_sha256(value)


def execution_state(model,ema,disc,optimizer,disc_optimizer):
    modules={'model':model,'ema':ema,'discriminator':disc}
    def names(module,opt):
        lookup={id(p):n for n,p in module.named_parameters()}
        return [[lookup[id(p)] for p in group['params']] for group in opt.param_groups]
    return {'module_modes':{key:{name:m.training for name,m in module.named_modules()} for key,module in modules.items()},
            'requires_grad':{key:{name:p.requires_grad for name,p in module.named_parameters()} for key,module in modules.items()},
            'optimizer_parameter_names':{'optimizer':names(model,optimizer),'discriminator_optimizer':names(disc,disc_optimizer)}}


def save_checkpoint(path,model,ema,disc,optimizer,disc_optimizer,scaler,binding,cursor,scheduler_state):
    path=Path(path)
    error=None
    try:
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Checkpoint must follow cleared gradients at optimizer boundary')
        if path.exists():raise FileExistsError('Refusing to replace retained checkpoint: '+str(path))
        local=rng_state(scaler)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    _collective_preflight(error)
    broadcast_buffers(disc)
    states=[None]*(dist.get_world_size() if distributed() else 1)
    if distributed():dist.all_gather_object(states,local,group=metadata_group())
    else:states[0]=local
    write_error=None
    if rank()==0:
        try:
            payload={'schema_version':'objective-paper-checkpoint-v1','binding':binding,'cursor':cursor,
                     'scheduler':scheduler_state,'world_size':len(states),'rank_states':states,
                     'model':model.state_dict(),'ema':ema.state_dict(),'discriminator':disc.state_dict(),
                     'optimizer':optimizer.state_dict(),'discriminator_optimizer':disc_optimizer.state_dict(),
                     'execution_state':execution_state(model,ema,disc,optimizer,disc_optimizer)}
            temp=path.with_name(path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)
            digest=file_sha256(path)
            seal={'sha256':digest,'bytes':path.stat().st_size,'binding_sha256':binding.get('binding_sha256')}
            if int(cursor.get('global_update',-1))==0:
                seal.update(initial_state_sha256=initial_state_sha256(payload),rank_rng_sha256=[state_value_sha256(state) for state in states])
            atomic_json(path.with_suffix(path.suffix+'.identity.json'),seal)
        except Exception as exc:write_error=f'{type(exc).__name__}: {exc}'
    errors=[write_error]
    if distributed():dist.broadcast_object_list(errors,src=0,group=metadata_group())
    if errors[0]:raise IOError('Collective checkpoint write failed: '+errors[0])
    return path


def load_checkpoint(path,model,ema,disc,optimizer,disc_optimizer,scaler,binding):
    saved=None;error=None;path=Path(path)
    try:
        seal=json.loads(path.with_suffix(path.suffix+'.identity.json').read_text())
        if file_sha256(path)!=seal['sha256']:raise ValueError('Checkpoint file checksum mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('schema_version')!='objective-paper-checkpoint-v1':raise ValueError('Unsupported resume schema')
        if saved['binding']!=binding:raise ValueError('Resume source/config/data identity mismatch')
        expected_world=dist.get_world_size() if distributed() else 1
        if saved['world_size']!=expected_world or len(saved['rank_states'])!=expected_world:raise ValueError('Exact resume requires same world size and rank states')
        if saved['cursor'].get('micro_step',0)!=0:raise ValueError('Only optimizer-boundary continuation is supported')
        current_execution=execution_state(model,ema,disc,optimizer,disc_optimizer)
        incoming_execution=saved['execution_state']
        if incoming_execution['optimizer_parameter_names']!=current_execution['optimizer_parameter_names']:
            raise ValueError('Optimizer parameter-name/group mapping changed')
        if incoming_execution['requires_grad']!=current_execution['requires_grad']:
            raise ValueError('Checkpoint trainable/frozen parameter roles changed')
        for name in ('model','ema','discriminator'):
            if incoming_execution['module_modes'][name].keys()!=current_execution['module_modes'][name].keys():
                raise ValueError('Checkpoint module-mode keys differ')
        for name,module in [('model',model),('ema',ema),('discriminator',disc)]:
            current=module.state_dict();incoming=saved[name]
            if current.keys()!=incoming.keys():raise ValueError(name+' state keys differ')
            if any(current[k].shape!=incoming[k].shape or current[k].dtype!=incoming[k].dtype for k in current):raise ValueError(name+' tensor schema differs')
        for name,opt in [('optimizer',optimizer),('discriminator_optimizer',disc_optimizer)]:
            current=opt.state_dict()['param_groups'];incoming=saved[name]['param_groups']
            if len(current)!=len(incoming) or any(len(a['params'])!=len(b['params']) for a,b in zip(current,incoming)):
                raise ValueError(name+' parameter groups differ')
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'

    _collective_preflight(error)
    mutation_error=None
    try:
        model.load_state_dict(saved['model'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
        disc.load_state_dict(saved['discriminator'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);disc_optimizer.load_state_dict(saved['discriminator_optimizer'])
        for name,module in [('model',model),('ema',ema),('discriminator',disc)]:
            for key,submodule in module.named_modules():submodule.training=saved['execution_state']['module_modes'][name][key]
        if saved['rank_states'][rank()].get('scaler') is not None:
            scaler.load_state_dict(saved['rank_states'][rank()]['scaler'])

        actual={'model':model.state_dict(),'ema':ema.state_dict(),'discriminator':disc.state_dict(),
                'optimizer':optimizer.state_dict(),'discriminator_optimizer':disc_optimizer.state_dict(),
                'execution_state':execution_state(model,ema,disc,optimizer,disc_optimizer)}
        for name,value in actual.items():
            if state_value_sha256(value)!=state_value_sha256(saved[name]):raise ValueError('Exact restored state differs: '+name)
        if state_value_sha256(scaler.state_dict() if scaler else None)!=state_value_sha256(saved['rank_states'][rank()]['scaler']):
            raise ValueError('Exact restored scaler differs')
    except Exception as exc:mutation_error=f'{type(exc).__name__}: {exc}'
    _collective_preflight(mutation_error)
    broadcast_buffers(disc)
    return saved['cursor'],saved['scheduler'],saved['rank_states'][rank()]
