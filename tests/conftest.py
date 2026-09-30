import torch

# the test models are tiny, so extra threads only add overhead (and slow down any training running alongside)
torch.set_num_threads(1)
