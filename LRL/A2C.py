from .utils import *
from .network_heads import *

def A2C(parclass):
  """Requires parent class, inherited from Agent."""
    
  class A2C(parclass):
    """
    Advantage Actor-Critic algorithm (A2C).
    Requires parent class inherited from Agent.
    Based on: https://arxiv.org/abs/1602.01783
    
    Args:
        FeatureExtractorNet - class inherited from nn.Module
        head - class of Actor-Critic network head, ActorCriticHead or SeparatedActorCriticHead
        rollout - number of frames for one iteration of updating NN weights
        noisy - use NoisyLinear instead of Linear layers if true, bool
        gamma - infinite horizon protection, float, from 0 to 1
        entropy_loss_weight - weight of additional entropy loss
        critic_loss_weight - weight of critic loss
        optimizer - class inherited from torch.optimizer, Adam by default
        optimizer_args - arguments for optimizer, dictionary
        grad_norm_max - maximum of gradient norm
    """
    __doc__ += parclass.__doc__
    
    def __init__(self, FeatureExtractor, head=ActorCriticHead, rollout=5, noisy=False, gamma=0.99, entropy_loss_weight=0.0, critic_loss_weight=1.0, 
                 optimizer=optim.Adam, optimizer_args={}, grad_norm_max=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.critic_loss_weight = critic_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.rollout = rollout
        self.noisy = noisy
        
        self.net = head(FeatureExtractor, self.noisy, self.env)
        self.net.after_init()
            
        self.optimizer = optimizer(self.net.parameters(), **optimizer_args)
        self.grad_norm_max = grad_norm_max

        self.observations = Tensor(size=(self.rollout + 1, self.env.num_envs, *self.env.observation_space.shape)).zero_()
        self.rewards = Tensor(size=(self.rollout, self.env.num_envs, 1)).zero_()
        self.actions = LongTensor(size=(self.rollout, self.env.num_envs, 1)).zero_()
        self.masks = Tensor(size=(self.rollout + 1, self.env.num_envs, 1)).zero_()
        self.returns = Tensor(size=(self.rollout + 1, self.env.num_envs, 1)).zero_()
        self.step = 0

        self.actor_loss_log = []
        self.critic_loss_log = []
        self.entropy_loss_log = []        
        if self.noisy:
            self.magnitude_log = []

    def act(self, s):
        with torch.no_grad():
            dist, values = self.net(Tensor(s))
            actions = dist.sample().view(-1, 1)

        return actions.view(-1).cpu().numpy()
    
    def see(self, state, action, reward, next_state, done, died):
        self.observations[self.step].copy_(Tensor(state))
        self.observations[self.step + 1].copy_(Tensor(next_state))
        self.actions[self.step].copy_(LongTensor(action).view(-1, 1))
        self.rewards[self.step].copy_(Tensor(reward).view(-1, 1))
        self.masks[self.step + 1].copy_(Tensor(1 - done).view(-1, 1))
        
        self.step = (self.step + 1) % self.rollout
        
        if self.step == 0:
            self.update()
            
            if self.noisy:
                self.magnitude_log.append(self.policy_net.magnitude())
            
    def update(self):
        """One step of optimization based on rollout memory"""
        self.net.train()
        
        with torch.no_grad():
            _, next_value = self.net(self.observations[-1])        

        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                self.gamma * self.masks[step + 1] + self.rewards[step]
        
        obs_shape = self.observations.size()[2:]
        action_shape = self.actions.size()[-1]
        num_steps, num_processes, _ = self.rewards.size()
        
        dist, values = self.net(self.observations[:-1].view(-1, *obs_shape))
        action_log_probs = dist.log_prob(self.actions.view(-1))
        dist_entropy = dist.entropy().mean()

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.returns[:-1] - values
        critic_loss = advantages.pow(2).mean()

        actor_loss = -(advantages.detach() * action_log_probs).mean()

        loss = actor_loss + self.critic_loss_weight * critic_loss - self.entropy_loss_weight * dist_entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_max)
        self.optimizer.step()
        
        self.actor_loss_log.append(actor_loss.item())
        self.critic_loss_log.append(critic_loss.item())
        self.entropy_loss_log.append(dist_entropy.item())
    
    def write(self, f):
        super().write(f)
        pickle.dump(self.actor_loss_log, f)
        pickle.dump(self.critic_loss_log, f)
        pickle.dump(self.entropy_loss_log, f)
        
    def read(self, f):
        super().read(f)
        self.actor_loss_log = pickle.load(f)
        self.critic_loss_log = pickle.load(f)
        self.entropy_loss_log = pickle.load(f)

    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.net.load_state_dict(torch.load(name + "-net"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.net.state_dict(), name + "-net")
  return A2C
