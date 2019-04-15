from .utils import *

def OUNoise(parclass):
  """Requires parent class, inherited from Agent."""
    
  class OUNoise(parclass):
    """
    Ornstein-Uhlenbeck noise process.
    
    Args:
        OU_mu - float
        OU_theta - float
        OU_sigma - float
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | {"OU_mu", "OU_theta", "OU_sigma"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("OU_mu", 0)
        self.config.setdefault("OU_theta", 0.15)
        self.config.setdefault("OU_sigma", 0.2)
        
        self.noise = self.config.OU_mu * np.ones(self.config.num_actions)

    def act(self, state):
        a = super().act(state, record)
        if self.is_learning:
            self.noise += self.config.OU_theta * (self.config.OU_mu - self.noise) + self.config.OU_sigma * np.random.normal(size=self.config.num_actions)
            return a + self.noise
        return a
  return OUNoise
