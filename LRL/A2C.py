from .utils import *
from .network_heads import *
       
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
        gamma - infinite horizon protection, float, from 0 to 1
        entropy_loss_weight - weight of additional entropy loss
        critic_loss_weight - weight of critic loss
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("ActorCritic") | {"ActorCriticHead", "rollout", "gamma", 
                                                             "entropy_loss_weight", "critic_loss_weight"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("ActorCriticHead", ActorCriticHead)
        self.config.setdefault("gamma", 0.99)
        self.config.setdefault("rollout", 5)
        self.config.setdefault("critic_loss_weight", 1)
        self.config.setdefault("entropy_loss_weight", 0)
        
        self.policy = self.config.ActorCriticHead(self.config, "ActorCritic").to(device)
        self.policy.init_optimizer()
        
        self.observations = Tensor(size=(self.config.rollout + 1, self.env.num_envs, *self.config.observation_shape)).zero_()
        self.rewards = Tensor(size=(self.config.rollout, self.env.num_envs)).zero_()
        self.actions = self.ActionTensor(size=(self.config.rollout + 1, self.env.num_envs, *self.config.actions_shape)).zero_()
        self.dones = Tensor(size=(self.config.rollout + 1, self.env.num_envs)).zero_()
        self.returns = Tensor(size=(self.config.rollout + 1, self.env.num_envs)).zero_()
        self.step = 0
        
        self.log_net(self.policy, "policy")
        self.logger_labels["actor_loss"] = ("training iteration", "loss")
        self.logger_labels["critic_loss"] = ("training iteration", "loss")
        self.logger_labels["entropy_loss"] = ("training iteration", "loss")

    def act(self, s, record=False):
        self.policy.eval()
        
        with torch.no_grad():
            dist, values = self.policy(Tensor(s))
            actions = dist.sample()
            
            if record:
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
    
    def compute_returns(self, values):
        '''
        Fills self.returns using self.rewards, self.dones
        input: values - Tensor, (self.rollout + 1, num_processes)
        '''
        self.returns[-1] = values[-1]
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * self.config.gamma * (1 - self.dones[step + 1]) + self.rewards[step]
    
    def update(self):
        """One step of optimization based on rollout memory"""
        self.policy.train()
        
        dist, values = self.policy(self.observations.view(-1, *self.config.observation_shape))
        action_log_probs = dist.log_prob(self.actions.view(-1, *self.config.actions_shape))

        values = values.view(self.config.rollout + 1, self.env.num_envs)
        action_log_probs = action_log_probs.view(self.config.rollout + 1, self.env.num_envs, -1)[:-1].sum(dim=-1)
        
        self.compute_returns(values)
        
        # calculating loss
        advantages = self.returns[:-1].detach() - values[:-1]
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean()
        entropy_loss = dist.entropy().view(self.config.rollout + 1, self.env.num_envs, -1)[:-1].sum(dim=-1).mean()
        
        loss = actor_loss + self.config.critic_loss_weight * critic_loss - self.config.entropy_loss_weight * entropy_loss
        
        # making a step of optimization
        self.policy.optimize(loss)
        
        # logging
        self.logger["actor_loss"].append(actor_loss.item())
        self.logger["critic_loss"].append(critic_loss.item())
        self.logger["entropy_loss"].append(entropy_loss.item())
        
    def show_record(self):
        show_frames_and_distribution(self.record["frames"], np.array(self.record["policies"])[:, 0:1], "Policy", np.arange(self.config["num_actions"]))
    
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.policy.load_state_dict(torch.load(name + "-net"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.policy.state_dict(), name + "-net")
  return A2C
