from .targetDQN import *

def DoubleQAgent(parclass):
  """Requires parent class, inherited from Agent.
  Already inherits from TargetQAgent"""
    
  class DoubleQAgent(TargetQAgent(parclass)):
    """
    Double DQN implementation.
    Based on: https://arxiv.org/abs/1509.06461
    """
    __doc__ += TargetQAgent(parclass).__doc__
                    
    def estimate_next_state(self, next_state_b):
        chosen_actions = self.q_net.greedy(self.q_net(next_state_b))
        return self.target_net.gather(self.target_net(next_state_b), chosen_actions)
  return DoubleQAgent
