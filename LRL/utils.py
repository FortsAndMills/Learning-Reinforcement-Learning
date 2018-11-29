import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

import pickle

from itertools import count
import random, math
import numpy as np
import time

import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

import gym
import gym.spaces        # to avoid warnings
gym.logger.set_level(40) # to avoid warnings

USE_CUDA = torch.cuda.is_available()
Tensor = lambda *args, **kwargs: torch.FloatTensor(*args, **kwargs).cuda(async=True) if USE_CUDA else torch.FloatTensor(*args, **kwargs)
LongTensor = lambda *args, **kwargs: torch.LongTensor(*args, **kwargs).cuda(async=True) if USE_CUDA else torch.LongTensor(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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
  
def show_frames_and_distribution(frames, distributions, name, support):
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
    plt.title(name)
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

def plot_durations(agent, means_window=100, points_limit=750000):
    """plot agent logs"""    
    clear_output(wait=True)    
    
    coords = [agent.logger_labels[key] for key in agent.logger.keys()]
    k = 0
    plots = {}
    for p in coords:
        if p not in plots:
            plots[p] = k; k += 1
            
    if len(plots) == 0:
        print("No logs in logger yet...")
        return
    
    plt.figure(2, figsize=(7.5 * ((len(plots) + 1) // 2), 7))
    plt.title('Training...')
    
    for i, plot_labels in enumerate(plots.keys()):
        plt.subplot((len(plots) + 1) // 2, 2, i + 1)
        plt.xlabel(plot_labels[0])
        plt.ylabel(plot_labels[1])        
    
    for key, value in agent.logger.items():
        plt.subplot((len(plots) + 1) // 2, 2, plots[agent.logger_labels[key]] + 1)
        plt.plot(value[-points_limit:])
        
        if key == "rewards":
            plt.plot(sliding_average(value, means_window)[-points_limit:])
        if key == "fps":
            plt.title("Current FPS: " + str(agent.logger["fps"][-1]))
    plt.show()
