from .utils import *
from .network_heads import *
       
class ActorCritic(Head):
    '''Actor-critic with shared feature extractor'''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.actor_head = self.linear(self.feature_size, config.num_actions)      
        self.critic_head = self.linear(self.feature_size, 1)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Categorical(logits=self.actor_head(features)), self.critic_head(features)
        
class SeparatedActorCritic(Head):
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

class FactorizedNormalActorCritic(Head):
    '''Actor-critic with shared feature extractor for continious action space'''
    '''Policy p(a | s) is approximated with factorized gaussian'''
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.actor_head_mu = self.linear(self.feature_size, config.num_actions)
        self.actor_head_sigma = nn.Sequential(
                            self.linear(self.feature_size, config.num_actions),
                            nn.Softplus()
                            )
        self.critic_head = self.linear(self.feature_size, 1)
        
    def forward(self, state):
        features = self.feature_extractor_net(state)
        return Normal(self.actor_head_mu(features), self.actor_head_sigma(features)), self.critic_head(features)

def A2C(parclass):
  """Requires parent class, inherited from Agent."""
    
  class A2C(parclass):
    """
    Advantage Actor-Critic algorithm (A2C).
    Requires parent class inherited from Agent.
    Based on: https://arxiv.org/abs/1602.01783
    
    Args:
        ActorCriticHead - class of Actor-Critic network head, ActorCriticHead or SeparatedActorCriticHead
        rollout - number of frames for one iteration of updating NN weights
        entropy_loss_weight - weight of additional entropy loss
        critic_loss_weight - weight of critic loss
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("ActorCritic") | {"ActorCriticHead", "rollout", 
                                                             "entropy_loss_weight", "critic_loss_weight"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("ActorCriticHead", ActorCritic)
        self.config.setdefault("rollout", 5)
        self.config.setdefault("critic_loss_weight", 1)
        self.config.setdefault("entropy_loss_weight", 0)
        self.config["value_repr_shape"] = ()
        
        self.policy = self.config.ActorCriticHead(self.config, "ActorCritic").to(device)
        self.policy.init_optimizer()
        
        self.observations = Tensor(size=(self.config.rollout + 1, self.env.num_envs, *self.config.observation_shape)).zero_()
        self.rewards = Tensor(size=(self.config.rollout, self.env.num_envs)).zero_()
        self.actions = self.ActionTensor(size=(self.config.rollout + 1, self.env.num_envs, *self.config.actions_shape)).zero_()
        self.dones = Tensor(size=(self.config.rollout + 1, self.env.num_envs)).zero_()
        self.step = 0
        
        self.logger_labels["actor_loss"] = ("training iteration", "loss")
        self.logger_labels["critic_loss"] = ("training iteration", "loss")
        self.logger_labels["entropy_loss"] = ("training iteration", "loss")

    def act(self, s):
        if self.is_learning:
            self.policy.train()
        else:
            self.policy.eval()
        
        with torch.no_grad():
            dist, values = self.policy(Tensor(s))
            actions = dist.sample()
            
            if self.is_recording:
                self.record["policies"].append(dist.probs.cpu().numpy())
                self.record["values"].append(values.cpu().numpy())

        return actions.cpu().numpy()
    
    def see(self, state, action, reward, next_state, done):
        super().see(state, action, reward, next_state, done)
        
        self.observations[self.step].copy_(Tensor(state))
        self.observations[self.step + 1].copy_(Tensor(next_state))
        self.actions[self.step].copy_(self.ActionTensor(action))
        self.rewards[self.step].copy_(Tensor(reward))
        self.dones[self.step + 1].copy_(Tensor(done.astype(np.float32)))
        
        self.step = (self.step + 1) % self.config.rollout        
        if self.step == 0:
            self.update()
    
    def compute_returns(self):
        '''
        Fills self.returns using self.values, self.rewards, self.dones
        '''
        d = len(self.config.value_repr_shape)
         
        self.returns[-1] = self.values[-1]
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * self.config.gamma * (1 - align(self.dones[step + 1], d)) + align(self.rewards[step], d)
            
    def preprocess_rollout(self):
        """Calculates action_dist, values, action_log_probs, returns based on current rollout"""
        self.policy.train()
        
        self.action_dist, self.values = self.policy(self.observations.view(-1, *self.config.observation_shape))
        self.action_log_probs = self.action_dist.log_prob(self.actions.view(-1, *self.config.actions_shape))#.sum(dim=-1)    
        
        self.values = self.values.view(self.config.rollout + 1, self.env.num_envs, *self.config.value_repr_shape)
        self.action_log_probs = self.action_log_probs.view(self.config.rollout + 1, self.env.num_envs)
        
        self.returns = torch.zeros_like(self.values)  # TODO: move to __init__?
        self.compute_returns()
        
    def optimized_function(self):
        """
        Returns value of optimized function for current batch:
        output: FloatTensor, (rollout, num_envs) 
        """ 
        advantages = self.returns_b - self.values_b
        return -(advantages.detach() * self.action_log_probs_b)
    
    def critic_loss(self):
        """
        Returns estimation of advantage for current batch:
        output: FloatTensor, (rollout, num_envs) 
        """
        return (self.returns_b.detach() - self.values_b).pow(2)
        
    def entropy_loss(self):
        """
        Returns entropy for current batch:
        output: FloatTensor, (rollout, num_envs) 
        """ 
        return -self.entropy_b
        
    def gradient_ascent_step(self):
        """Makes one update of policy weights"""
        
        # calculating loss        
        actor_loss = self.optimized_function().mean()
        critic_loss = self.critic_loss().mean()        
        entropy_loss = self.entropy_loss().mean()
        
        loss = actor_loss + self.config.critic_loss_weight * critic_loss + self.config.entropy_loss_weight * entropy_loss
        
        # making a step of optimization
        self.policy.optimize(loss)
        
        # logging
        self.logger["actor_loss"].append(actor_loss.item())
        self.logger["critic_loss"].append(self.config.critic_loss_weight * critic_loss.item())
        self.logger["entropy_loss"].append(self.config.entropy_loss_weight * entropy_loss.item())
    
    def update(self):
        """One step of optimization based on rollout memory"""
        self.preprocess_rollout()
        
        # in basic A2C algorithm, batch is constructed using all rollout.
        # we do not use the last states from environments as we do not know return for it yet.
        self.returns_b = self.returns[:-1].view(-1)
        self.values_b = self.values[:-1].view(-1)
        self.action_log_probs_b = self.action_log_probs[:-1].view(-1)    
        self.entropy_b = self.action_dist.entropy()[:-1].view(-1)    #TODO .sum(dim=-1) inside?
        
        self.gradient_ascent_step()        
        
    def show_record(self):
        show_frames_and_distribution(self.record["frames"], np.array(self.record["policies"])[:, 0:1], "Policy", np.arange(self.config["num_actions"]))
    
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.policy.load_state_dict(torch.load(name + "-net"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.policy.state_dict(), name + "-net")
  return A2C
