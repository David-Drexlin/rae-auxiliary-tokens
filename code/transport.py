import os

def checkpoint_runtime():
    return {'NCCL_P2P_DISABLE': os.environ.get('NCCL_P2P_DISABLE'), 'NCCL_IB_DISABLE': os.environ.get('NCCL_IB_DISABLE')}
