from .utils import *

import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython.display import display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

def show_frames(frames):
    """
    generate animation inline notebook:
    input: frames - list of pictures
    """      
    
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    
    # not working in matplotlib 3.1
    display(display_animation(anim, default_mode='loop'))
  
def show_frames_and_distribution(frames, distributions, name, support):
    """
    generate animation inline notebook with distribtuions plot
    input: frames - list of pictures
    input: distributions - list of arrays of fixed size
    input: name - title name
    input: support - indexes for support of distribution
    """ 
         
    plt.figure(figsize=(frames[0].shape[1] / 34.0, frames[0].shape[0] / 72.0), dpi = 72)
    plt.subplot(121)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    plt.subplot(122)
    plt.title(name)
    action_patches = []
    for a in range(distributions.shape[1]):
        action_patches.append(plt.bar(support, distributions[0][a], width=support[1]-support[0]))

    def animate(i):
        patch.set_data(frames[i])
        
        for a, action_patch in enumerate(action_patches): 
            for rect, yi in zip(action_patch, distributions[i][a]):
                rect.set_height(yi)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames) - 1, interval=50)
    
    # not working in matplotlib 3.1
    display(display_animation(anim, default_mode='loop'))
    
def sliding_average(a, window_size):
    """one-liner for sliding average for array a with window size window_size"""
    return np.convolve(np.concatenate([np.ones((window_size - 1)) * a[0], a]), np.ones((window_size))/window_size, mode='valid')

def plot_durations(agent, means_window=100, points_limit=1000):
    """
    plot agent logs;
    input: means_window - size of sliding window for rewards
    input: points_limit - maximal number of points to visualise on each plot 
    """    
    clear_output(wait=True)    
    
    # getting what plots do we want to draw
    coords = [agent.logger_labels[key] for key in agent.logger.keys() if key in agent.logger_labels]
    k = 0
    plots = {}
    for p in coords:
        if p not in plots:
            plots[p] = k; k += 1
            
    if len(plots) == 0:
        print("No logs in logger yet...")
        return
    
    plt.figure(2, figsize=(11, 4 * ((len(plots) + 1) // 2)))
    plt.title('Training...')
    
    # creating plots
    axes = []
    for i, plot_labels in enumerate(plots.keys()):
        axes.append(plt.subplot((len(plots) + 1) // 2, 2, i + 1))
        plt.xlabel(plot_labels[0])
        plt.ylabel(plot_labels[1])        
    
    for key, value in agent.logger.items():
        if key in agent.logger_labels:
            # id of plot in which we want to draw a line
            ax = axes[plots[agent.logger_labels[key]]]

            # we do not want to draw many points
            k = len(value) // points_limit + 1
            ax.plot(np.arange(len(value))[::k], value[::k], label=key)  # TODO: averaging instead?
            ax.legend()
            
            # smoothing main plot!
            if key == "rewards":
                ax.plot(np.arange(len(value))[::k], sliding_average(value, means_window)[::k], label="smoothed rewards")
    plt.show()
