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
                           
    def copy_configuration(config, source, destination):
        for suffix in ["_FeatureExtractor",
                       "_linear", 
                       "_optimizer", 
                       "_optimizer_args", 
                       "_clip_gradients"]:
            if source + suffix in config:
                config[destination + suffix] = config[source + suffix]
    
    def __init__(self, config, name):
        super().__init__()
        
        self.name = name
        self.config = config
        self.linear = config.get(name + "_linear", nn.Linear)
        self.feature_extractor_net = config[name + "_FeatureExtractor"](self.linear).to(device)
        self.feature_size = self.get_feature_size()
        self.optimization_steps_done = 0
        
        self.config.logger_labels[name + "_magnitude"] = ("magnitude logging iteration", "noise magnitude")
        self.config.logger_labels[name + "_loss"] = ("training iteration", "loss")
    
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
        
        self.optimization_steps_done += 1
        self.log(loss)
    
    def get_feature_size(self):
        '''
        Returns feature size of self.feature_extractor_net
        output: feature_size - int
        '''
        return self.feature_extractor_net(Tensor(size=(10, *self.config["observation_shape"]))).size()[1]
        
    def average_magnitude(self):
        '''
        Returns average magnitude of the whole net
        output: float
        '''
        mag, n_params = sum([np.array(layer.magnitude()) for layer in self.modules() if hasattr(layer, "magnitude")])
        return mag / n_params
        
    def log(self, loss):
        '''
        Adds to logs information about this net
        input: loss - Tensor, (1)
        '''
        self.config.logger[self.name + "_loss"].append(loss.detach().cpu().numpy())
        
        if self.linear is not nn.Linear:
            if (self.optimization_steps_done % self.config.magnitude_logging_fraction == 0):
                self.config.logger[self.name + "_magnitude"].append(self.average_magnitude())
        
    def forward(self, state):
        raise NotImplementedError()
        
    def numel(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
