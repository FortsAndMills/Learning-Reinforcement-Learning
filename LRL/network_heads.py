from .utils import *
from .network_modules import *

class QnetworkHead(nn.Module):
    '''Abstract class for all q-network heads'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env):
        super().__init__()
        
        self.linear = NoisyLinear if noisy else nn.Linear
        self.feature_extractor_net = feature_extractor_net(self.linear)
        self.feature_size = feature_size
        self.num_actions = env.action_space.n
        
    def after_init(self):
        '''called after initialisation'''
        if USE_CUDA:
            self = self.cuda()
            
        self.noisy_layers = []
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                self.noisy_layers.append(m)
    
    def greedy(self, output):
        '''returns greedy action based on output of net'''
        return output.max(1)[1]
        
    def gather(self, output, action_b):
        '''returns output of net for given batch of actions'''
        return output.gather(1, action_b.unsqueeze(1)).squeeze(1)
    
    def value(self, output):
        '''returns output for greedily-chosen action'''
        return output.max(1)[0]
        
    def magnitude(self):
        '''returns average magnitude of the whole net'''
        mag, n_params = sum([np.array(noisy_layer.magnitude()) for noisy_layer in self.noisy_layers])
        return mag / n_params
        
class Vnetwork(QnetworkHead):
    '''V-network head supposing env has next_states function'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env): 
        super().__init__(feature_extractor_net, feature_size, noisy, env)       
        self.head = nn.Linear(self.feature_size, 1)
        self.next_states = env.next_states_function
        
    def forward(self, state):
        next_states, r, done = self.next_states(state)
        next_states = next_states.view(state.size()[0] * self.num_actions, -1)
        
        next_states = self.feature_extractor_net(next_states)
        return r + (1 - done) * self.head(next_states).view(state.size()[0], self.num_actions)

class Qnetwork(QnetworkHead):
    '''Simple Q-network head'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env): 
        super().__init__(feature_extractor_net, feature_size, noisy, env)       
        self.head = nn.Linear(self.feature_size, self.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state)
        
class DuelingQnetwork(QnetworkHead):
    '''Dueling version of Q-network head'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env): 
        super().__init__(feature_extractor_net, feature_size, noisy, env)       
        self.v_head = nn.Linear(feature_size, 1)
        self.a_head = nn.Linear(feature_size, self.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state)
        a = self.a_head(state)
        return v + a - a.mean(dim=1, keepdim=True)
        
class CategoricalQnetworkHead(QnetworkHead):
    '''Abstract class for q-network heads for categorical DQN'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env, num_atoms, support):
        super().__init__(feature_extractor_net, feature_size, noisy, env)
        self.num_atoms = num_atoms
        self.support = support
    
    def greedy(self, output):
        return (output * self.support).sum(2).max(1)[1]
    
    def gather(self, output, action_b):
        return output.gather(1, action_b.unsqueeze(1).unsqueeze(1).expand(output.size(0), 1, output.size(2))).squeeze(1)
    
    def value(self, output):
        return self.gather(output, self.greedy(output))
               
class CategoricalQnetwork(CategoricalQnetworkHead):
    '''Simple categorical DQN head'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env, num_atoms, support):
        super().__init__(feature_extractor_net, feature_size, noisy, env, num_atoms, support)       
        self.head = nn.Linear(feature_size, self.num_actions * self.num_atoms) 
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return F.softmax(self.head(state).view(-1, self.num_actions, self.num_atoms), dim=-1)
        
class DuelingCategoricalQnetwork(CategoricalQnetworkHead):
    '''Dueling version of categorical DQN head'''
    def __init__(self, feature_extractor_net, feature_size, noisy, env, num_atoms, support):
        super().__init__(feature_extractor_net, feature_size, noisy, env, num_atoms, support)    
        self.v_head = nn.Linear(feature_size, self.num_atoms)
        self.a_head = nn.Linear(feature_size, self.num_actions * self.num_atoms)    
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state).view(-1, 1, self.num_atoms)
        a = self.a_head(state).view(-1, self.num_actions, self.num_atoms)
        outp = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(outp, dim=-1)
