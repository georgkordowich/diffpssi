"""
File that defines the backend for the simulation models.
Torch enables GPU acceleration, and gradient tracking. Numpy is faster, but does not support GPU acceleration or
gradient tracking. Note: Even if torch is used, GPU is not always faster than CPU.
"""
BACKEND = 'torch'  # 'torch' or 'numpy'
DEFAULT_DEVICE = 'cpu'  # 'cuda' or 'cpu'

if BACKEND == 'torch':
    import torch
    print('Using torch as backend')
    if torch.cuda.is_available() and (DEFAULT_DEVICE == 'cuda' or DEFAULT_DEVICE == 'gpu'):
        torch.set_default_device('cuda')
        print('Using torch with CUDA on GPU')
    else:
        print('Using torch on CPU')

elif BACKEND == 'numpy':
    import numpy as torch
    torch.tensor = torch.array
    torch.unsqueeze = torch.expand_dims
    print('Using numpy as backend')
else:
    raise ValueError('No backend specified')
