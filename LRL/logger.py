from .utils import *
from .network_modules import *
from collections import defaultdict

class Logger():
    """
    Util agent interface including logging
    
    Args:    
        magnitude_logging_fraction - number of optimization steps between magnitude logging (it's expensive to do each iteration), int
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
        
        self.config.setdefault("magnitude_logging_fraction", 20)
        self.config["logger"] = self.logger
        self.config["logger_labels"] = self.logger_labels
    
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
