from .DQN import *

class TDEHead(QnetworkHead):
    '''
    Output format: two FloatTensors of size (batch_size, num_actions)
    First is Q*(s, a), second is TD_error(s, a)
    '''
    def sample(self, output):
        return Tensor(output[0].shape).uniform_(0, 1) * output[1] + output[0]

    def greedy(self, output):
        return self.sample(output).max(1)[1]
        
    def gather(self, output, action_b):
        return output[0].gather(1, action_b.unsqueeze(1)).squeeze(1), \
               output[1].gather(1, action_b.unsqueeze(1)).squeeze(1)
    
    def value(self, output):
        return self.sample(output).max(1)[0]

class TDEnetwork(TDEHead):
    '''Simple Q-network head'''
    def __init__(self, config, name): 
        super().__init__(config, name)      
        self.head = self.linear(self.feature_size, config.num_actions)
        self.error_head = self.linear(self.feature_size, config.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state), self.error_head(state)

def TDEAgent(parclass):
  """
  Requires parent class, inherited from Agent.
  Already inherits from QAgent.
  """
    
  class TDEAgent(QAgent(parclass)):
    """
    EXPERIMENTAL
    """
    __doc__ += parclass.__doc__
    def __init__(self, config):
        config.setdefault("QnetworkHead", TDEnetwork)
        super().__init__(config)
        
        self.logger_labels["td_error loss"] = ("training iteration", "loss")
    
    def get_loss(self, guess, q):
        '''
        Calculates batch loss
        input: guess - target, FloatTensor, (batch_size)
        input: q - current model output, tuple:
                q[0] is Q*(s, a), FloatTensor, (batch_size, num_actions)
                q[1] is TDE(s, a), FloatTensor, (batch_size, num_actions)
        output: FloatTensor, (batch_size)
        '''
        assert guess.shape == q[0].shape
        assert guess.shape == q[1].shape
        tderror = guess - q[0]
        self.logger["td_error loss"].append(abs(tderror).mean())
        return tderror.pow(2) + (tderror - q[1]).pow(2)
  return TDEAgent
