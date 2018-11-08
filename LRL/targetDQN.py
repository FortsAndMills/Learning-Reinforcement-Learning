from .DQN import *

def TargetQAgent(parclass):
  """Requires parent class, inherited from Agent"""   
    
  class TargetQAgent(parclass):
    '''
    Target network heuristic implementation.
    
    Args:
        target_update - frequency in frames of updating target network
    '''
    __doc__ += parclass.__doc__
    
    def __init__(self, config):
        super().__init__(config)

        self.target_net = self.init_network() 
        self.unfreeze()

        self.target_update = config.get("target_update", 100)

    def unfreeze(self):
        '''copy policy net weights to target net'''
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def see(self, state, action, reward, next_state, done):
        super().see(state, action, reward, next_state, done)

        if self.frames_done % self.target_update == 0:
            self.unfreeze()
    
    def estimate_next_state(self, next_state_b):
        return self.target_net.value(self.target_net(next_state_b))

    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.unfreeze()
  return TargetQAgent
