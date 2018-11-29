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

# HEADS FOR DQN ---------------------------------------------------------------------------------        

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

class Qnetwork(QnetworkHead):
    '''Simple Q-network head'''
    def __init__(self, config, name): 
        super().__init__(config, name)      
        self.head = self.linear(self.feature_size, config.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state)
        
class DuelingQnetwork(QnetworkHead):
    '''Dueling version of Q-network head'''
    def __init__(self, config, name): 
        super().__init__(config, name)       
        self.v_head = self.linear(self.feature_size, 1)
        self.a_head = self.linear(self.feature_size, config.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state)
        a = self.a_head(state)
        return v + a - a.mean(dim=1, keepdim=True)

# HEADS FOR Categorical DQN ---------------------------------------------------------------------------------

class CategoricalQnetworkHead(QnetworkHead):
    '''Abstract class for q-network heads for categorical DQN'''
        
    def greedy(self, output):
        return (output * self.config.support).sum(2).max(1)[1]
    
    def gather(self, output, action_b):
        return output.gather(1, action_b.unsqueeze(1).unsqueeze(1).expand(output.size(0), 1, output.size(2))).squeeze(1)
    
    def value(self, output):
        return self.gather(output, self.greedy(output))

class CategoricalQnetwork(CategoricalQnetworkHead):
    '''Simple categorical DQN head'''
    def __init__(self, config, name):
        super().__init__(config, name)       
        self.head = self.linear(self.feature_size, config.num_actions * config.num_atoms) 
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return F.softmax(self.head(state).view(-1, self.config.num_actions, self.config.num_atoms), dim=-1)
        
class DuelingCategoricalQnetwork(CategoricalQnetworkHead):
    '''Dueling version of categorical DQN head'''
    def __init__(self, config, name):
        super().__init__(config, name)    
        self.v_head = self.linear(self.feature_size, config.num_atoms)
        self.a_head = self.linear(self.feature_size, config.num_actions * config.num_atoms)    
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state).view(-1, 1, self.config.num_atoms)
        a = self.a_head(state).view(-1, self.config.num_actions, self.config.num_atoms)
        outp = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(outp, dim=-1)

# HEADS FOR DDPG ---------------------------------------------------------------------------------

class DDPG_Actor(Head):
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.head = nn.Sequential(
            self.linear(self.feature_size, config.num_actions),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.head(self.feature_extractor_net(state)).view(-1, *self.config.actions_shape)

class DDPG_Critic(QnetworkHead):
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.head = self.linear(self.feature_size, 1)
        
    def get_feature_size(self):
        return self.feature_extractor_net(Tensor(size=(10, *self.config["observation_shape"])),
                                          Tensor(size=(10, *self.config["actions_shape"]))).size()[1]
        
    def forward(self, state):
        return state
    
    def greedy(self, output):
        return self.config["actor"](output)
        
    def gather(self, output, action_b):
        return self.head(self.feature_extractor_net(output, action_b)).squeeze(dim=1)
    
    def value(self, output):
        return self.head(self.feature_extractor_net(output, self.greedy(output))).squeeze(dim=1)
        
# HEADS FOR ACTOR-CRITIC ---------------------------------------------------------------------------------
        
class ActorCriticHead(Head):
    '''Actor-critic with shared feature extractor'''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.actor_head = self.linear(self.feature_size, config.num_actions)      
        self.critic_head = self.linear(self.feature_size, 1)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Categorical(logits=self.actor_head(features)), self.critic_head(features)
        
class SeparatedActorCriticHead(Head):
    '''Separate two nets for actor-critic '''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.head = self.linear(self.feature_size, config.num_actions)
        
        self.critic = nn.Sequential(
            config[name + "_FeatureExtractor"](self.linear),
            self.linear(self.feature_size, 1)
        )
        
    def forward(self, state):
        value = self.critic(state)
        probs = self.head(self.feature_extractor_net(state))
        return Categorical(logits=probs), value

class FactorizedNormalActorCriticHead(Head):
    '''Actor-critic with shared feature extractor for continious action space'''
    '''Policy p(a | s) is approximated with factorized gaussian'''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.actor_head_mu = self.linear(self.feature_size, config.num_actions)
        self.actor_head_sigma = self.linear(self.feature_size, config.num_actions)      
        self.critic_head = self.linear(self.feature_size, 1)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Normal(self.actor_head_mu(features), self.actor_head_sigma(features)**2), self.critic_head(features)

# TO DEL
# TODO this was made for Rubik's cube and doesn't follow library conventions...
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
