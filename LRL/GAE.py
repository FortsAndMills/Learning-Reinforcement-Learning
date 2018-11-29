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
    PARAMS = A2C(parclass).PARAMS | {"gae_tau"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("gae_tau", 0.95)
        assert self.config.gae_tau > 0 and self.config.gae_tau <= 1, "Gae Tau must lie in (0, 1]"
    
    def compute_returns(self, values):
        gae = 0
        for step in reversed(range(self.rewards.size(0))): # just some arithmetics ;)
            delta = self.rewards[step] + self.config.gamma * values[step + 1] * (1 - self.dones[step + 1]) - values[step]
            gae = delta + self.config.gamma * self.config.gae_tau * (1 - self.dones[step + 1]) * gae
            self.returns[step] = gae + values[step]
  return GAE
