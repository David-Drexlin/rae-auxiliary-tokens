import math
import torch
import torch.distributed as dist

def gradient_diagnostics(image_loss,aux_loss,shared_parameters,distributed=False):






    parameters=[p for p in shared_parameters if p.requires_grad]
    if not parameters:raise ValueError('A shared trainable parameter set is required')
    vectors=[]
    for loss in (image_loss,aux_loss):
        grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(parameters)
        vector=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for p,g in zip(parameters,grads)])
        if not torch.isfinite(vector).all():raise RuntimeError('Nonfinite shared gradients')
        if distributed:
            if not dist.is_initialized():raise ValueError('Distributed gradient probe requires initialized process group')
            dist.all_reduce(vector);vector/=dist.get_world_size()
        vectors.append(vector)
    norms=[float(v.norm()) for v in vectors]
    denominator=norms[0]*norms[1]
    return {'shared_image_gradient_norm':norms[0],'shared_auxiliary_gradient_norm':norms[1],
            'shared_gradient_cosine':float(torch.dot(*vectors))/denominator if denominator else None,
            'aux_to_image_gradient_norm':norms[1]/norms[0] if norms[0] else None,
            'undefined_cosine':denominator==0,'parameter_elements':vectors[0].numel(),
            'gradient_scope':'rank_mean_microstep' if distributed else 'local_microstep',
            'gradient_accumulation_included':False}


def dct_matrix(size,device):
    n=torch.arange(size,dtype=torch.float64,device=device)
    k=n[:,None]
    matrix=torch.cos(math.pi/size*(n+.5)*k)*math.sqrt(2/size)
    matrix[0]/=math.sqrt(2)
    return matrix


def image_error_metrics(reconstruction,reference,include_dct=False):

    if reconstruction.shape!=reference.shape or reconstruction.ndim!=4 or reconstruction.shape[1]!=3:
        raise ValueError('Matched BCHW RGB arrays required')
    error=reconstruction.double()-reference.double()
    if not torch.isfinite(error).all():raise ValueError('Nonfinite image error')
    mean=error.mean(dim=(-2,-1),keepdim=True)
    total=error.square().mean(dim=(1,2,3))
    mean_error=mean.square().mean(dim=(1,2,3))
    centered=(error-mean).square().mean(dim=(1,2,3))
    residual=total-mean_error-centered
    if float(residual.abs().max())>1e-10:raise AssertionError('Total MSE does not equal channel-mean plus centered MSE')
    result={'rgb_l1':error.abs().mean(dim=(1,2,3)),'total_rgb_mse':total,'channel_mean_rgb_error':mean_error,'centered_rgb_mse':centered,
            'centered_identity_residual':residual}
    if include_dct:
        height,width=error.shape[-2:]
        if height!=width or height%32:raise ValueError('Frozen radial DCT bands require square size divisible by32')
        matrix=dct_matrix(height,error.device)
        coefficients=torch.matmul(torch.matmul(matrix,error),matrix.T)
        coordinates=torch.arange(height,device=error.device,dtype=torch.int64)
        radius2=coordinates[:,None].square()+coordinates[None,:].square()
        masks={'dc':radius2==0,'very_low':(radius2>0)&(radius2*1024<=height*height),
               'low':(radius2*1024>height*height)&(radius2*64<=height*height),
               'mid':(radius2*64>height*height)&(radius2*16<=height*height),
               'high':radius2*16>height*height}
        contributions=[]
        for name,mask in masks.items():
            count=int(mask.sum())
            coefficient_mean=coefficients[:,:,mask].square().mean(dim=(1,2))
            contribution=coefficient_mean*(count/(height*height))
            result[f'dct_{name}_coefficient_mse']=coefficient_mean
            result[f'dct_{name}_pixel_mse']=contribution
            result[f'dct_{name}_cardinality']=count
            contributions.append(contribution)
        result['dct_total_mse']=sum(contributions)
        result['dct_non_dc_mse']=sum(contributions[1:])
        for actual,expected,label in [(sum(contributions),total,'total'),(contributions[0],mean_error,'DC'),(sum(contributions[1:]),centered,'non-DC')]:
            if not torch.allclose(actual,expected,rtol=1e-9,atol=1e-10):raise AssertionError(f'DCT {label} identity failed')
    return result
