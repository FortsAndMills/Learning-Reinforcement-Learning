from .DQN import *

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

def CategoricalQAgent(parclass):
  """Requires parent class, inherited from Agent.
  Already inherits from QAgent"""
    
  class CategoricalQAgent(QAgent(parclass)):
    """
    Categorical DQN.
    Based on: https://arxiv.org/pdf/1707.06887.pdf
    
    Args:
        Vmin - minimum value of approximation distribution, int
        Vmax - maximum value of approximation distribution, int
        num_atoms - number of atoms in approximation distribution, int
    """
    __doc__ += QAgent(parclass).__doc__
    PARAMS = QAgent(parclass).PARAMS | {"Vmin", "Vmax", "num_atoms"} 
    
    def __init__(self, config):
        config.setdefault("Vmin", -10)
        config.setdefault("Vmax", 10)
        config.setdefault("num_atoms", 51)        
        
        config.setdefault("QnetworkHead", CategoricalQnetwork)
        
        assert config["Vmin"] < config["Vmax"], "Vmin must be less than Vmax!"
        assert issubclass(config["QnetworkHead"], CategoricalQnetworkHead)
         
        super().__init__(config)
        
        self.support = torch.linspace(self.config.Vmin, self.config.Vmax, self.config.num_atoms).to(device)
        self.config["support"] = self.support
        
        self.offset = torch.linspace(0, (self.config.batch_size - 1) * self.config.num_atoms, self.config.batch_size).long().unsqueeze(1).expand(self.config.batch_size, self.config.num_atoms).to(device)
            
    def batch_target(self, reward_b, next_state_b, done_b):
        delta_z = float(self.config.Vmax - self.config.Vmin) / (self.config.num_atoms - 1)
        
        next_dist = self.estimate_next_state(next_state_b)

        reward_b = reward_b.unsqueeze(1).expand_as(next_dist)
        done_b   = done_b.unsqueeze(1).expand_as(next_dist)
        support = self.support.unsqueeze(0).expand_as(next_dist)

        Tz = reward_b + (1 - done_b) * (self.config.gamma**self.config.replay_buffer_nsteps) * support
        Tz = Tz.clamp(min=self.config.Vmin, max=self.config.Vmax)
        b  = (Tz - self.config.Vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()        
        
        proj_dist = Tensor(next_dist.size()).zero_()              
        proj_dist.view(-1).index_add_(0, (l + self.offset).view(-1), (next_dist * (u.float()+ (b.ceil() == b).float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + self.offset).view(-1), (next_dist * (b - l.float())).view(-1))
        proj_dist /= proj_dist.sum(1).unsqueeze(1)
        return proj_dist

    def get_loss(self, y, guess):
        guess.data.clamp_(0.0001, 0.9999)   # TODO doesn't torch have cross entropy? Taken from source code.
        return -(y * guess.log()).sum(1)
        
    def get_transition_importance(self, loss_b):
        return loss_b
        
    def show_record(self):
        show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"])[:, 0], "Future reward distribution", self.support.cpu().numpy())
        
  return CategoricalQAgent
        
