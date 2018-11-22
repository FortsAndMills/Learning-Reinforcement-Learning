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
    
    def __init__(self, config):
        super().__init__(config)
        
        self.epsilon_by_frame = lambda frame_idx: config.get("epsilon_final", 0.01) + (config.get("epsilon_start", 1) - config.get("epsilon_final", 0.01)) * math.exp(-1. * frame_idx / config.get("epsilon_decay", 500))

    def act(self, state):
        if self.is_learning:
            explore = np.random.uniform(0, 1, size=state.shape[0]) <= self.epsilon_by_frame(self.frames_done)
            
            actions = np.zeros((state.shape[0]), dtype=int)
            actions[explore] = [self.env.action_space.sample() for _ in range(explore.sum())]   # TODO vectorize?
            if (~explore).sum() > 0:
                actions[~explore] = super().act(state[~explore])
            return actions
        else:
            return super().act(state)
  return eGreedy
