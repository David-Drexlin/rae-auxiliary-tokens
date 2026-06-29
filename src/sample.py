# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch.nn as nn
import math
from time import time
import argparse
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from stage2.state_utils import (
    decode_stage2_state,
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    infer_aux_state_spec,
    make_initial_sample_state,
    split_guided_state,
)
from utils.train_utils import parse_configs
from stage1 import RAE
from torchvision.utils import save_image
import torch
import sys
import os
from stage2.models import Stage2ModelProtocol
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        _data_config,
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc,
        _training_config,
        _eval_config,
    ) = parse_configs(args.config)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()  # important!
    rae.eval()
    aux_state_spec = infer_aux_state_spec(rae, latent_size=tuple(misc.get("latent_size", (768, 16, 16))))
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(
        shift_dim / shift_base)
    print(
        f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")
    
    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = misc.get("latent_size", (768, 16, 16))
    # Labels to condition the model with (feel free to change):
    class_labels = [0, 1] if num_classes <= 2 else [0, min(1, num_classes - 1)]

    # Create sampling noise:
    n = len(class_labels)
    state = make_initial_sample_state(
        n=n,
        latent_size=tuple(latent_size),
        aux_state_spec=aux_state_spec,
        device=torch.device(device),
    )
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    state = duplicate_state_for_guidance(state)
    y_null = torch.tensor([null_label] * n, device=device)
    y = torch.cat([y, y_null], 0)
    
    # set guidance setup
    guidance_scale = guidance_config.get("scale", 1.0)
    if guidance_scale > 1.0:
        t_min, t_max = guidance_config.get("t_min", 0.0), guidance_config.get("t_max", 1.0)
        model_kwargs = dict(y=y, cfg_scale=guidance_scale,
                            cfg_interval=(t_min, t_max))
        guidance_method = guidance_config.get("method", "cfg")
        if guidance_method == "autoguidance":
            guid_model_config = guidance_config.get("guidance_model", None)
            assert guid_model_config is not None, "Please provide a guidance model config when using autoguidance."
            guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
            guid_model.eval()  # important!
            guid_fwd = guid_model.forward
            model_kwargs['additional_model_forward'] = guid_fwd
            model_fwd = model.forward_with_autoguidance
        else:
            model_fwd = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fwd = model.forward
    # Sample images:
    start_time = time()
    samples = final_state_from_trajectory(sample_fn(state, model_fwd, **model_kwargs))
    samples = split_guided_state(samples)  # Remove null class samples
    samples = decode_stage2_state(rae, samples)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_known_args()[0]
    main(args)
