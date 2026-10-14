import copy

def matched_decoder(original,config,num_patches):
 import torch
 if config.decoder_aux_mode!='prepend' or hasattr(config,'decoder_parameter_footing_num_aux_tokens'):return original(config,num_patches)
 ref=copy.deepcopy(config);ref.decoder_aux_mode='discard';ref.num_aux_tokens=0
 with torch.random.fork_rng(devices=[]):reference=original(ref,num_patches)
 native=original(config,num_patches);current=native.state_dict()
 with torch.no_grad():
  for name,value in reference.state_dict().items():
   if name not in current or current[name].shape!=value.shape:raise ValueError('Shared decoder parameter mismatch')
   current[name].copy_(value)
 return native
