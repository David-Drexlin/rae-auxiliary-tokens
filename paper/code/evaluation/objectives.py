import torch
from torch import nn

class ReconstructionObjective(nn.Module):
    def __init__(self, rae, config):
        super().__init__()
        self.rae = rae
        self.config = dict(config)
        self.kind = 'none'
        self.matching_weight = 0.0
        self.regularization_weight = 0.0
        self.active = False
        self.heads = nn.ModuleDict()
        self.locations = {}
        self.rae.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.rae.encoder.eval()
        return self

    def forward(self, images, include_aux=True):
        reconstruction = self.rae(images)
        return {'reconstruction': reconstruction, 'auxiliary_loss': reconstruction.new_zeros((), dtype=torch.float32), 'components': {}}

def build_objective(rae, config):
    if config.get('objective_kind') != 'none':
        raise ValueError('This decoder uses the image reconstruction objective')
    return ReconstructionObjective(rae, config)
