import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import pickle

from itertools import count
import random, math
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

USE_CUDA = torch.cuda.is_available()
Tensor = lambda *args, **kwargs: torch.cuda.FloatTensor(*args, **kwargs) if USE_CUDA else torch.FloatTensor(*args, **kwargs)
LongTensor = lambda *args, **kwargs: torch.cuda.LongTensor(*args, **kwargs) if USE_CUDA else torch.LongTensor(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def show_frames(frames):
    """
    generate animation inline notebook:
    frames - list of pictures
    """      
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))
  
def show_frames_and_distribution(frames, distributions, support):
    """
    generate animation inline notebook with distribtuions plot
    frames - list of pictures
    distributions - list of arrays of fixed size
    support - indexes for support of distribution
    """      
    plt.figure(figsize=(frames[0].shape[1] / 34.0, frames[0].shape[0] / 72.0), dpi = 72)
    plt.subplot(121)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    plt.subplot(122)
    action_patches = []
    for a in range(distributions.shape[1]):
        action_patches.append(plt.bar(support, distributions[0][a]))

    def animate(i):
        patch.set_data(frames[i])
        
        for a, action_patch in enumerate(action_patches): 
            for rect, yi in zip(action_patch, distributions[i][a]):
                rect.set_height(yi)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames) - 1, interval=50)
    display(display_animation(anim, default_mode='loop'))

def sliding_average(a, window_size):
    """one-liner for sliding average for array a with window size window_size"""
    return np.convolve(np.concatenate([np.ones((window_size - 1)) * a[0], a]), np.ones((window_size))/window_size, mode='valid')

def plot_durations(agent, means_window=100):
    """plot agent logs"""    
    clear_output(wait=True)
    
    plt.figure(2, figsize=(15, 7))
    plt.title('Training...')
    
    plt.subplot(221)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if len(agent.rewards_log) > 0:
        plt.plot(agent.rewards_log)
        plt.plot(sliding_average(agent.rewards_log, means_window))
    
    if hasattr(agent, 'loss_log'):
        plt.subplot(222)
        plt.xlabel('Update iteration')
        plt.ylabel('Loss')
        plt.plot(agent.loss_log)
        
    if hasattr(agent, 'critic_loss_log'):
        plt.subplot(222)
        plt.xlabel('Update iteration')
        plt.ylabel('Loss')
        plt.plot(agent.critic_loss_log, label="critic")
        plt.plot(agent.actor_loss_log, label="actor")
        plt.legend()
        
    if hasattr(agent, 'magnitude_log'):
        plt.subplot(223)
        plt.xlabel('Update iteration')
        plt.ylabel('Noise Average Magnitude')
        plt.plot(agent.magnitude_log)
    plt.show()
