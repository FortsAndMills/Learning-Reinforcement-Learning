from .DQN import *

def TwinQAgent(parclass):
  """Requires parent class, inherited from Agent."""
    
  class TwinQAgent(parclass):
    """
    Twin DQN (two DQN run in parallel).
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("TwinQnetwork") 
    
    def __init__(self, config):
        super().__init__(config)
        Head.copy_configuration(self.config, "Qnetwork", "TwinQnetwork")
              
        self.twin_q_net = self.config.QnetworkHead(self.config, "TwinQnetwork").to(device)
        self.twin_q_net.init_optimizer()
    
    def estimate_next_state(self, next_state_b):
        return self.twin_q_net.value(self.q_net(next_state_b))
    
    def optimize_model(self):
        super().optimize_model()
        self.twin_q_net, self.q_net = self.q_net, self.twin_q_net        
        super().optimize_model()
        self.twin_q_net, self.q_net = self.q_net, self.twin_q_net
        
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.twin_q_net.load_state_dict(torch.load(name + "-twin_qnet"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.twin_q_net.state_dict(), name + "-twin_qnet")
  return TwinQAgent
