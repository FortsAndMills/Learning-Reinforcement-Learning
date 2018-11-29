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
    PARAMS = parclass.PARAMS | {"epsilon_start", "epsilon_final", "epsilon_decay"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("epsilon_final", 0.01)
        self.config.setdefault("epsilon_start", 1)
        self.config.setdefault("epsilon_decay", 500)
        
        self.epsilon_by_frame = lambda frame_idx: self.config.epsilon_final + (self.config.epsilon_start - self.config.epsilon_final) * \
                                                                               math.exp(-1. * frame_idx / self.config.epsilon_decay)

    def act(self, state, record=False):
        if self.is_learning:
            explore = np.random.uniform(0, 1, size=state.shape[0]) <= self.epsilon_by_frame(self.frames_done)
            
            actions = np.zeros((state.shape[0], *self.config.actions_shape))
            if explore.sum() > 0:
                actions[explore] = np.array([self.env.action_space.sample() for _ in range(explore.sum())])
            if (~explore).sum() > 0:
                actions[~explore] = super().act(state[~explore])
            return actions
        else:
            return super().act(state, record)
  return eGreedy
