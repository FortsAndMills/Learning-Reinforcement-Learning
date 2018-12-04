from .utils import *
from .network_modules import *

class Head(nn.Module):
    '''
    Abstract class for network heads
    
    Args:
    FeatureExtractor - class of feature extractor neural net
    linear - class for linear layers in network: nn.Linear or NoisyLinear
    '''
    
    PARAMS = lambda name: {name + "_FeatureExtractor", 
                           name + "_linear", 
                           name + "_optimizer", 
                           name + "_optimizer_args", 
                           name + "_clip_gradients"}   
    
    def __init__(self, config, name):
        super().__init__()
        
        self.name = name
        self.config = config
        self.linear = config.get(name + "_linear", nn.Linear)
        self.feature_extractor_net = config[name + "_FeatureExtractor"](self.linear).to(device)
        self.feature_size = self.get_feature_size()
    
    def init_optimizer(self):
        '''initialize optimizer for this network'''
        self.clip_gradients = self.config.get(self.name + "_clip_gradients", None)
        self.optimizer = self.config.get(self.name + "_optimizer", optim.Adam)(self.parameters(), **self.config.get(self.name + "_optimizer_args", {}))
    
    def optimize(self, loss):
        '''
        Make one step of stochastic gradient optimization
        input: loss - Tensor, (1)
        '''
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradients)
        self.optimizer.step()
    
    def get_feature_size(self):
        '''
        Returns feature size of self.feature_extractor_net
        output: feature_size - int
        '''
        return self.feature_extractor_net(Tensor(size=(10, *self.config["observation_shape"]))).size()[1]
        
    def forward(self, state):
        raise NotImplementedError()
    
    def greedy(self, output):
        '''
        Returns greedy action based on output of net
        input: output - FloatTensor (in format of this head's forward function output)
        output: LongTensor, (batch_size)
        '''
        raise NotImplementedError()
        
    def gather(self, output, action_b):
        '''
        Returns output of net for given batch of actions
        input: output - FloatTensor (in format of this head's forward function output)
        input: action_b - LongTensor (batch_size)
        output: FloatTensor, head's forward function output for selected actions
        '''
        raise NotImplementedError()
    
    def value(self, output):
        '''
        Returns value of action, chosen greedy
        input: output - FloatTensor (in format of this head's forward function output)
        output: FloatTensor, (batch_size)
        '''
        raise NotImplementedError()
