from contextlib import nullcontext
import torch
import torch.nn.functional as F
from state import broadcast_buffers,frozen_evaluation,update_ema
from preservation_backward import preservation_backward


def _step_count(optimizer,parameter):
    value=optimizer.state.get(parameter,{}).get('step',0)
    return float(value.item() if torch.is_tensor(value) else value)


def train_update(*,model,ddp_model,discriminator,ddp_discriminator,optimizer,
                 disc_optimizer,ema,scaler,microbatches,image_loss_fn,augment,
                 device,precision,epoch,base_loss,training,last_layer,
                 diagnostics_fn=None,teacher=None,verification_callback=None):






    verification=None
    if verification_callback is not None:
        microbatches=tuple(microbatches)
        verification={'model':model,'ddp_model':ddp_model,'discriminator':discriminator,'ddp_discriminator':ddp_discriminator,
            'optimizer':optimizer,'disc_optimizer':disc_optimizer,'ema':ema,'scaler':scaler,'microbatches':microbatches,
            'image_loss_fn':image_loss_fn,'augment':augment,'device':device,'precision':precision,'epoch':epoch,
            'base_loss':base_loss,'training':training,'last_layer':last_layer,'diagnostics_fn':diagnostics_fn,'teacher':teacher}
        verification_callback('before_update',verification)
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
    def autocast():return torch.autocast(device.type,dtype=dtype,enabled=precision!='fp32')
    accum=int(training.get('grad_accum_steps',1))
    use_gan=epoch>=int(base_loss.get('adversarial_start_epoch',8)) and float(base_loss.get('adversarial_weight',.75))>0
    train_disc=epoch>=int(base_loss.get('disc_update_start_epoch',6)) and float(base_loss.get('adversarial_weight',.75))>0
    optimizer.zero_grad(set_to_none=True);disc_optimizer.zero_grad(set_to_none=True)
    broadcast_buffers(discriminator)
    logs={};diagnostics={};last_images=None;seen=0
    for micro,entry in enumerate(microbatches):
        if micro>=accum:raise ValueError('Too many microbatches')
        images=entry[0] if isinstance(entry,(tuple,list)) else entry
        images=images.to(device,non_blocking=True);last_images=images;seen+=1
        sync=ddp_model.no_sync() if hasattr(ddp_model,'no_sync') and micro<accum-1 else nullcontext()
        with sync:
            with frozen_evaluation(discriminator),autocast():
                result=ddp_model(images,include_aux=True)
                reconstruction=result['reconstruction'];aux=result['auxiliary_loss']
                image,components=image_loss_fn(images,reconstruction)
                gan=image.new_zeros(())
                if use_gan:


                    logits,_=discriminator(augment.aug(2*reconstruction-1),None)
                    gan=-logits.mean()
                if diagnostics_fn is not None and micro==accum-1:
                    with preservation_backward(model,device,phase='auxiliary_gradient_diagnostics'):
                        diagnostics=diagnostics_fn(image,aux)
            if use_gan:
                reconstruction_gradient=torch.autograd.grad(image,last_layer,retain_graph=True)[0].float().norm()
                adversarial_gradient=torch.autograd.grad(gan,last_layer,retain_graph=True)[0].float().norm()
                adaptive=(reconstruction_gradient/(adversarial_gradient+1e-6)).clamp(0,float(base_loss.get('max_adaptive_weight',1000))).detach()
            else:adaptive=image.new_zeros(())
            loss=image+aux+float(base_loss.get('adversarial_weight',.75))*adaptive*gan
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite generator loss')
            with preservation_backward(model,device,phase='generator_loss_backward'):
                scaler.scale(loss.float()/accum).backward()
        for name,value in {**components,**result.get('components',{}),'auxiliary_total':aux,'gan':gan,'adaptive_weight':adaptive,'total':loss}.items():
            if torch.is_tensor(value):logs[name]=logs.get(name,0)+float(value.detach())/accum
    if seen!=accum:raise ValueError('Incomplete accumulated optimizer batch')
    if teacher is not None and any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Frozen teacher acquired a gradient')
    trainable=[p for group in optimizer.param_groups for p in group['params']]
    scaler.unscale_(optimizer)
    clip=float(training.get('clip_grad',0))
    if clip>0:torch.nn.utils.clip_grad_norm_(trainable,clip)
    elif not scaler.is_enabled():torch.nn.utils.clip_grad_norm_(trainable,float('inf'),error_if_nonfinite=True)
    counted=next((p for p in trainable if p.grad is not None),None)
    if counted is None:raise RuntimeError('Generator has no trainable gradients')
    if verification_callback is not None:verification_callback('generator_gradients',verification)
    before=_step_count(optimizer,counted);scaler.step(optimizer);generator_updated=_step_count(optimizer,counted)>before
    optimizer.zero_grad(set_to_none=True)
    if train_disc:
        prior=model.training;model.eval()
        with torch.no_grad(),autocast():fresh=model(last_images,include_aux=False)['reconstruction']
        model.train(prior);discriminator.train();disc_optimizer.zero_grad(set_to_none=True)
        fake=(2*fresh-1).clamp(-1,1);fake=torch.round((fake+1)*127.5)/127.5-1
        with autocast():
            fake_logits,real_logits=ddp_discriminator(augment.aug(fake),augment.aug(2*last_images-1))
            disc_loss=.5*(F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean())
        if not torch.isfinite(disc_loss):raise FloatingPointError('Nonfinite discriminator loss')
        scaler.scale(disc_loss.float()).backward()
        scaler.unscale_(disc_optimizer)
        if not scaler.is_enabled():
            torch.nn.utils.clip_grad_norm_([p for p in discriminator.parameters() if p.requires_grad],float('inf'),error_if_nonfinite=True)
        if verification_callback is not None:verification_callback('discriminator_gradients',verification)
        scaler.step(disc_optimizer);disc_optimizer.zero_grad(set_to_none=True)
        broadcast_buffers(discriminator);discriminator.eval()
        logs['discriminator_hinge']=float(disc_loss.detach())
    scaler.update()
    if generator_updated:update_ema(ema,model,float(training.get('ema_decay',.9978)))
    logs['generator_optimizer_updated']=generator_updated
    logs['scaler_scale']=float(scaler.get_scale())
    result={'components':logs,'diagnostics':diagnostics,'last_images':last_images}
    if verification_callback is not None:verification_callback('update_complete',{**verification,'result':result,'last_images':last_images})
    return result
