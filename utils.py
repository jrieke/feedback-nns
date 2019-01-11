import random
import numpy as np
import torch

def seed_all(seed=None):
    """Set seed for random, numpy.random, torch and torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        torch.manual_seed(np.random.randint(1e6))
        torch.cuda.manual_seed(np.random.randint(1e6))
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
