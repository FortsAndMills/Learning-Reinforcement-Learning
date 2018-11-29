from .utils import *
from .network_modules import *
from collections import defaultdict

class Logger():
    """
    Util agent interface including logging
    
    Args:    
        magnitude_logging_fraction - number of frames between magnitude logging (it's expensive to do each iteration), int
    """
    
    def __init__(self, config):
        
        # checking for wrong parameter names to avoid serious problems:        
        self.config = AttrDict(config)
        for key in self.config:
            if key not in type(self).PARAMS:
                raise Exception("Unknown config parameter: " + key)
        
        # logging
        self.logger = defaultdict(list)
        self.logger_labels = defaultdict(tuple)
        self.networks_to_log = []
        
        self.config.setdefault("magnitude_logging_fraction", 1000)
    
    def log_net(self, network, name):
        '''
        Adds network to the list of networks to log
        input: network - an instance of nn.Module
        input: name - string
        '''
        self.networks_to_log.append((network, name))
        if self.config.get("linear_layer", nn.Linear) is NoisyLinear:  # TODO this check for noisyness is outdated! Doesn't work!
            self.logger_labels[name + "_magnitude"] = ("training epoch", "noise magnitude")        
    
    def magnitude(self, network):
        '''
        Returns average magnitude of the whole net
        input: network - an instance of nn.Module
        '''
        mag, n_params = sum([np.array(noisy_layer.magnitude()) for layer in self.network.modules() if isinstance(layer, NoisyLinear)])
        return mag / n_params
        
    def log(self):
        '''Called every training step'''
        if self.config.get("linear_layer", nn.Linear) is NoisyLinear:
            if (self.frames_done % self.config.magnitude_logging_fraction < self.env.num_envs):
                for network, name in self.networks_to_log:
                    self.logger[name + "_magnitude"].append(self.magnitude(network))
    
    # saving and loading functions
    def write(self, f):
        """writing logs and data to file f"""
        pickle.dump(self.logger, f)
        
    def read(self, f):
        """reading logs and data from file f"""
        self.logger = pickle.load(f)
        
    def save(self, name):
        """saving to file"""
        f = open(name, 'wb')
        self.write(f)
        f.close()   
        
    def load(self, name):
        """loading from file"""
        f = open(name, 'rb')
        self.read(f)
        f.close()
