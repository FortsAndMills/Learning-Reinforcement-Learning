from . import A2C

def GAE(parclass):
  """Requires parent class, inherited from Agent.
  Already inherits from A2C"""
    
  class GAE(A2C(parclass)):
    """
    Generalized Advantage Estimation (GAE) upgrade of A2C.
    Based on: https://arxiv.org/abs/1506.02438
    
    Args:
        gae_tau - float, from 0 to 1
    """
    __doc__ += A2C(parclass).__doc__
    
    def __init__(self, config):
        super().__init__(config)
        self.gae_tau = config.get("gae_tau", 0.95)
    
    def compute_returns(self, values):
        gae = 0
        for step in reversed(range(self.rewards.size(0))): # just some arithmetics ;)
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step + 1]) - values[step]
            gae = delta + self.gamma * self.gae_tau * (1 - self.dones[step + 1]) * gae
            self.returns[step] = gae + values[step]
  return GAE
