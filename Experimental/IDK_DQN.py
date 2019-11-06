from .DQN import *
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class IDKnetworkHead(QnetworkHead):
    def greedy(self, output):
        return output.sample().max(1)[1]
        
    def gather(self, output, action_b):
        return Normal(
                output.loc.gather(1, action_b.unsqueeze(1)).squeeze(1),
                output.scale.gather(1, action_b.unsqueeze(1)).squeeze(1))
    
    def value(self, output):
        return self.gather(output, self.greedy(output))

class IDKnetwork(IDKnetworkHead):
    def __init__(self, config, name): 
        super().__init__(config, name)      
        self.mu_head = self.linear(self.feature_size, config.num_actions)
        self.sigma_head = nn.Sequential(
            self.linear(self.feature_size, config.num_actions),
            nn.Softplus()
        )
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return Normal(self.mu_head(state), self.sigma_head(state) + 1e-4)

def IDKAgent(parclass):
  """
  Requires parent class, inherited from Agent.
  Already inherits from QAgent
  """
    
  class IDKAgent(QAgent(parclass)):
    """
    Experimental!
    """
    __doc__ += QAgent(parclass).__doc__
    PARAMS = QAgent(parclass).PARAMS | {"init_std", "energy_samples"} 
    
    def __init__(self, config):
        super().__init__(config)        
        self.config.setdefault("init_std", 1)
        self.config.setdefault("energy_samples", 2)

        #self.init_IDKnetwork()
        self.logger_labels["average_Q_std"] = ("training iteration", "average_Q_std")
        self.logger_labels["average_Q_mean"] = ("training iteration", "average_Q_mean")
        self.logger_labels["max_state_importance"] = ("training iteration", "max_state_importance")
    
    def batch_target(self, reward_b, next_state_b, done_b):
        '''
        Calculates target for batch to learn
        input: reward_batch - FloatTensor, (batch_size)
        input: next_state_batch - FloatTensor, (batch_size x state_dim)
        input: done_batch - FloatTensor, (batch_size)
        output: Normal distribution, (batch_size)
        '''
        next_q_values = self.estimate_next_state(next_state_b)
        return Normal(
            reward_b + (self.config.gamma**self.config.replay_buffer_nsteps) * next_q_values.loc * (1 - done_b),
            ((self.config.gamma**self.config.replay_buffer_nsteps)**0.5) * next_q_values.scale * (1 - done_b))

    def energy_distance(self, P, Q):
        '''
        Calculates energy distance
        input: P, Q - Normal distribution, (batch_size)
        '''
        x1 = P.rsample(sample_shape=(self.config.energy_samples,)).T
        x2 = P.rsample(sample_shape=(self.config.energy_samples,)).T
        y1 = Q.rsample(sample_shape=(self.config.energy_samples,)).T
        y2 = Q.rsample(sample_shape=(self.config.energy_samples,)).T

        return (torch.abs(x1[:, None] - y1[:, :, None]).mean(dim=1).mean(dim=1) +
                torch.abs(x2[:, None] - y2[:, :, None]).mean(dim=1).mean(dim=1) - 
                torch.abs(x1[:, None] - x2[:, :, None]).mean(dim=1).mean(dim=1) - 
                torch.abs(y1[:, None] - y2[:, :, None]).mean(dim=1).mean(dim=1))

    def get_loss(self, guess, q):
        '''
        Calculates batch loss
        input: guess - target, Normal distribution, (batch_size)
        input: q - current model output, Normal distribution, (batch_size)
        output: FloatTensor, (batch_size)
        '''
        self.logger["max_state_importance"].append(torch.max(q.loc.max(dim=-1)[0] - q.loc.min(dim=-1)[0])[0].item())
        self.logger["average_Q_mean"].append(q.loc.mean().item())
        self.logger["average_Q_std"].append(q.scale.mean().item())
        
        # Wasserstein distance: biased gradients
        #return (guess.loc - q.loc).pow(2) + (torch.clamp(guess.scale, min=1e-4) - q.scale).pow(2)
        # 
        # Cramer distance!
        return self.energy_distance(guess, q)   
    
  return IDKAgent
