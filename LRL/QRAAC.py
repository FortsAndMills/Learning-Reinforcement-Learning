from .A2C import *
       
class QuantileActorCritic(Head):
    '''Actor-critic with shared feature extractor'''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.actor_head = self.linear(self.feature_size, config.num_actions)      
        self.critic_head = self.linear(self.feature_size, config.quantiles)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Categorical(logits=self.actor_head(features)), self.critic_head(features)

def QRAAC(parclass):
  """
  Requires parent class, inherited from Agent.
  Already inherits from A2C
  """
    
  class QRAAC(A2C(parclass)):
    """
    Experimental!
    Quantile Regression Advantage Actor-Critic algorithm (QRAAC).
    Requires parent class inherited from Agent.
    
    Args:
        quantiles - number of atoms in approximation distribution, int
    """
    __doc__ += A2C(parclass).__doc__
    PARAMS = A2C(parclass).PARAMS | {"quantiles"}  
    
    def __init__(self, config):
        config.setdefault("quantiles", 51)       
        super().__init__(config)
        self.config["value_repr_shape"] = (self.config.quantiles,)
    
    def advantage_estimation(self):
        return self.returns_b.detach().mean(dim=-1) - self.values_b.mean(dim=-1)
    
    def critic_loss(self):
        tau = Tensor((2 * np.arange(self.config.quantiles) + 1) / (2.0 * self.config.quantiles))         
        diff = self.returns_b.detach().t()[:, :, None] - self.values_b[None]        
        return (diff * (tau.view(1, -1) - (diff < 0).float())).transpose(0,1).mean(1).sum(-1)
    
  return QRAAC
