from .utils import *

def eGreedy(parclass):
  """Requires parent class, inherited from Agent."""
    
  class eGreedy(parclass):
    """
    Basic e-Greedy exploration strategy.
    
    Args:
        epsilon_start - value of epsilon at the beginning, float, from 0 to 1
        epsilon_final - minimal value of epsilon, float, from 0 to 1
        epsilon_decay - degree of exponential damping of epsilon, int
    """
    __doc__ += parclass.__doc__
    
    def __init__(self, epsilon_start = 1.0, epsilon_final = 0.01, epsilon_decay = 500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    def act(self, state):
        if not self.learn or random.random() > self.epsilon_by_frame(self.frames_done):
            return super().act(state)
        else:
            return self.random_act()
  return eGreedy
