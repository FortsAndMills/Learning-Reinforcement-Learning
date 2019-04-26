from .DQN import *

class QuantileQnetworkHead(QnetworkHead):
    '''Abstract class for q-network heads for Quantile Regression DQN'''
        
    def greedy(self, output):
        return output.mean(2).max(1)[1]
    
    def gather(self, output, action_b):
        return output.gather(1, action_b.unsqueeze(1).unsqueeze(1).expand(output.size(0), 1, output.size(2))).squeeze(1)
    
    def value(self, output):
        return self.gather(output, self.greedy(output))

class QuantileQnetwork(QuantileQnetworkHead):
    '''Simple categorical DQN head'''
    def __init__(self, config, name):
        super().__init__(config, name)       
        self.head = self.linear(self.feature_size, config.num_actions * config.quantiles) 
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state).view(-1, self.config.num_actions, self.config.quantiles)
        
class DuelingQuantileQnetwork(QuantileQnetworkHead):
    '''Dueling version of categorical DQN head'''
    def __init__(self, config, name):
        super().__init__(config, name)    
        self.v_head = self.linear(self.feature_size, self.config.quantiles)
        self.a_head = self.linear(self.feature_size, self.config.num_actions * self.config.quantiles)    
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state).view(-1, 1, self.config.quantiles)
        a = self.a_head(state).view(-1, self.config.num_actions, self.config.quantiles)
        return v + a - a.mean(dim=1, keepdim=True)

def QuantileQAgent(parclass):
  """Requires parent class, inherited from Agent.
  Already inherits from QAgent"""
    
  class QuantileQAgent(QAgent(parclass)):
    """
    Quantile Regression DQN.
    Based on: https://arxiv.org/abs/1710.10044
    
    Args:
        quantiles - number of atoms in approximation distribution, int
    """
    __doc__ += QAgent(parclass).__doc__
    PARAMS = QAgent(parclass).PARAMS | {"quantiles"} 
    
    def __init__(self, config):
        config.setdefault("quantiles", 51)       
        config.setdefault("QnetworkHead", QuantileQnetwork)
        
        assert issubclass(config["QnetworkHead"], QuantileQnetworkHead)
         
        super().__init__(config)
        
    def batch_target(self, reward_b, next_state_b, done_b):
        next_q_values = self.estimate_next_state(next_state_b)
        return reward_b[:, None] + (self.config.gamma**self.config.replay_buffer_nsteps) * next_q_values * (1 - done_b)[:, None]  # TODO: just add extend_like to original
            
    def get_loss(self, guess, q):
        '''
        Calculates batch loss
        input: guess - target, FloatTensor, (batch_size, quantiles)
        input: q - current model output, FloatTensor, (batch_size, quantiles)
        output: FloatTensor, (batch_size)
        '''
        tau = Tensor((2 * np.arange(self.config.quantiles) + 1) / (2.0 * self.config.quantiles))         
        diff = guess.t()[:, :, None] - q[None]        
        return (diff * (tau.view(1, -1) - (diff < 0).float())).transpose(0,1).mean(1).sum(-1)
        
    def get_transition_importance(self, loss_b):
        return loss_b
        
    def show_record(self):
        # TODO: Arseny Ashuha's plot style
        show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"])[:, 0], "Future reward quantiles", np.linspace(0, 1, self.config.quantiles))
        
  return QuantileQAgent
        
