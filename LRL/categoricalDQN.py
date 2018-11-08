from .DQN import *

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
    
    def __init__(self, config):
        self.Vmin = config.get("Vmin", -10)
        self.Vmax = config.get("Vmax", 10)
        self.num_atoms = config.get("num_atoms", 51)
        
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)
        config["support"] = self.support      
        super().__init__(config)
    
    def batch_target(self, reward_b, next_state_b, done_b):
        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        
        next_dist = self.estimate_next_state(next_state_b)

        reward_b = reward_b.unsqueeze(1).expand_as(next_dist)
        done_b   = done_b.unsqueeze(1).expand_as(next_dist)
        support = self.support.unsqueeze(0).expand_as(next_dist)

        Tz = reward_b + (1 - done_b) * (self.gamma**self.replay_buffer_nsteps) * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b  = (Tz - self.Vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()
        
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms)
        proj_dist = torch.zeros(next_dist.size())
        if USE_CUDA:
            offset = offset.cuda()
            proj_dist = proj_dist.cuda()     
              
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float()+ (b.ceil() == b).float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        proj_dist /= proj_dist.sum(1).unsqueeze(1)
        return proj_dist

    def get_loss(self, y, guess):
        guess.data.clamp_(0.0001, 0.9999)   # TODO doesn't torch have cross entropy? Taken from source code.
        return -(y * guess.log()).sum(1).mean()
        
    def get_transition_importance(self, loss_b):
        return loss_b
        
    def show_record(self):
        show_frames_and_distribution(self.frames, np.array(self.record["qualities"])[:, 0], self.support.cpu().numpy())
        
  return CategoricalQAgent
        
