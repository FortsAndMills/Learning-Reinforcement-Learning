import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

from itertools import count
import random, math
import numpy as np
import pickle
import time

import gym
import gym.spaces        # to avoid warnings
gym.logger.set_level(40) # to avoid warnings

USE_CUDA = torch.cuda.is_available()
Tensor = lambda *args, **kwargs: torch.FloatTensor(*args, **kwargs).cuda() if USE_CUDA else torch.FloatTensor(*args, **kwargs)
LongTensor = lambda *args, **kwargs: torch.LongTensor(*args, **kwargs).cuda() if USE_CUDA else torch.LongTensor(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def align(tensor, i):
    """
    Adds i singleton dimensions to the end of tensor 
    """
    for _ in range(i):
        tensor = tensor[:, None]
    return tensor

