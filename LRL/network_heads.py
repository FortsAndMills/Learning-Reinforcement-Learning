from .utils import *
from .network_modules import *

class Head(nn.Module):
    '''
    Abstract class for network heads
    
    Args:
    env - environment with action_space and observation_space defined
    feature_extractor - class of feature extractor neural net
    feature_size - size of feature representation outputed by extractor, int
    linear_layer - class for linear layers in network: nn.Linear of NoisyLinear
    linear_layer_args - additional arguments for linear_layers (for example, std_init for NoisyLinear)    
    linear_layer_init - initialization for linear layers 
    '''
    
    def __init__(self, config):
        super().__init__()
        
        class SelectedLinear(config.get("linear_layer", nn.Linear)):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, **config.get("linear_layer_args", {}))
                
                if "linear_layer_init" in config:
                    config["linear_layer_init"](self)
        
        self.linear = SelectedLinear
        self.feature_extractor_net = config["FeatureExtractor"](self.linear).to(device)
        
        self.num_actions = config["num_actions"]
        if "feature_size" not in config:
            config["feature_size"] = self.feature_extractor_net(Tensor(size=(1, *config["observation_shape"]))).size()[1]
        self.feature_size = config["feature_size"]
        
    def after_init(self):
        '''must be called after initialisation'''
        if USE_CUDA:
            self = self.cuda()
            
        self.noisy_layers = []
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                self.noisy_layers.append(m)
    
    def greedy(self, output):
        '''returns greedy action based on output of net'''
        raise NotImplementedError()
        
    def gather(self, output, action_b):
        '''returns output of net for given batch of actions'''
        raise NotImplementedError()
    
    def value(self, output):
        '''returns output for greedily-chosen action'''
        raise NotImplementedError()
        
    def forward(self, state):
        raise NotImplementedError()
        
    def magnitude(self):
        '''returns average magnitude of the whole net'''
        mag, n_params = sum([np.array(noisy_layer.magnitude()) for noisy_layer in self.noisy_layers])
        return mag / n_params
        
class QnetworkHead(Head):
    def greedy(self, output):
        '''returns greedy action based on output of net'''
        return output.max(1)[1]
        
    def gather(self, output, action_b):
        '''returns output of net for given batch of actions'''
        return output.gather(1, action_b.unsqueeze(1)).squeeze(1)
    
    def value(self, output):
        '''returns output for greedily-chosen action'''
        return output.max(1)[0]
        
class Vnetwork(QnetworkHead):
    '''V-network head supposing env has next_states function'''
    def __init__(self, config): 
        super().__init__(env, config)
            
        self.head = self.linear(self.feature_size, 1)
        try:
            self.next_states = env.next_states_function
        except:
            raise Exception("Environment must have next_states_function which by batch of PyTorch states returns all possible next states")
        
    def forward(self, state):
        # emulate all possible next states
        next_states, r, done = self.next_states(state)
        next_states = next_states.view(state.size()[0] * self.num_actions, -1)
        
        next_states = self.feature_extractor_net(next_states)
        return r + (1 - done) * self.head(next_states).view(state.size()[0], self.num_actions)

class Qnetwork(QnetworkHead):
    '''Simple Q-network head'''
    def __init__(self, config): 
        super().__init__(config)      
        self.head = nn.Linear(self.feature_size, self.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state)
        
class DuelingQnetwork(QnetworkHead):
    '''Dueling version of Q-network head'''
    def __init__(self, config): 
        super().__init__(config)       
        self.v_head = nn.Linear(self.feature_size, 1)
        self.a_head = nn.Linear(self.feature_size, self.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state)
        a = self.a_head(state)
        return v + a - a.mean(dim=1, keepdim=True)
        
class CategoricalQnetworkHead(QnetworkHead):
    '''Abstract class for q-network heads for categorical DQN'''
    def __init__(self, config):
        super().__init__(config)
        self.num_atoms = config["num_atoms"]
        self.support = config["support"]
    
    def greedy(self, output):
        return (output * self.support).sum(2).max(1)[1]
    
    def gather(self, output, action_b):
        return output.gather(1, action_b.unsqueeze(1).unsqueeze(1).expand(output.size(0), 1, output.size(2))).squeeze(1)
    
    def value(self, output):
        return self.gather(output, self.greedy(output))

class CategoricalQnetwork(CategoricalQnetworkHead):
    '''Simple categorical DQN head'''
    def __init__(self, config):
        super().__init__(config)       
        self.head = nn.Linear(self.feature_size, self.num_actions * self.num_atoms) 
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return F.softmax(self.head(state).view(-1, self.num_actions, self.num_atoms), dim=-1)
        
class DuelingCategoricalQnetwork(CategoricalQnetworkHead):
    '''Dueling version of categorical DQN head'''
    def __init__(self, config):
        super().__init__(config)    
        self.v_head = nn.Linear(self.feature_size, self.num_atoms)
        self.a_head = nn.Linear(self.feature_size, self.num_actions * self.num_atoms)    
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state).view(-1, 1, self.num_atoms)
        a = self.a_head(state).view(-1, self.num_actions, self.num_atoms)
        outp = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(outp, dim=-1)
        
class ActorCriticHead(Head):
    '''Separate two nets for actor-critic '''
    def __init__(self, config):
        super().__init__(config)
        
        self.actor_head = self.linear(self.feature_size, self.num_actions)      
        self.critic_head = self.linear(self.feature_size, 1)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Categorical(logits=self.actor_head(features)), self.critic_head(features)
        
class SeparatedActorCriticHead(Head):
    '''Separate two nets for actor-critic '''
    def __init__(self, config):
        super().__init__(config)
        
        self.head = self.linear(self.feature_size, self.num_actions)
        
        self.critic = nn.Sequential(
            config["FeatureExtractor"](self.linear),
            self.linear(self.feature_size, 1)
        )
        
    def forward(self, state):
        value = self.critic(state)
        probs = self.head(self.feature_extractor_net(state))
        return Categorical(logits=probs), value
