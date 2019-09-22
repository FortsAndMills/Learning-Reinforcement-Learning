from .utils import *
from .network_heads import *

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
       
def PPO(parclass):
  """
  Requires parent class, inherited from A2C.
  """
  class PPO(parclass):
    """
    Proximal Policy Optimization algorithm (PPO).
    Requires parent class inherited from A2C.
    Based on: https://arxiv.org/abs/1707.06347
    
    Args:
        ppo_clip - clipping rate of pi_new / pi_old fraction
        epochs - number of epochs to run through rollout on each update
        batch_size - size of mini-batch to select without replacement on each gradient ascent step
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | {"ppo_clip", "epochs", "batch_size"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("ppo_clip", 0.2)
        self.config.setdefault("epochs", 3)
        self.config.setdefault("batch_size", 32)
        
        assert self.env.num_envs * self.config.rollout >= self.config.batch_size, "batch_size is bigger than rollout * number of threads!"
    
    def optimized_function(self):
        # advantage is estimated using advantage function for previous policy
        # so we take advantage from original rollout
        advantages = (self.returns_b - self.old_values_b).detach()  #self.advantages_b.detach() - КОСТЫЛЬ
        
        # importance sampling for making an update of current policy using samples from old policy
        # the gradients to policy will flow through the numerator.
        ratio = torch.exp(self.action_log_probs_b - self.old_action_log_probs_b.detach())
        
        # PPO clipping! Prevents from "too high updates".
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip) * advantages
        return -torch.min(surr1, surr2)
       
    def update(self):
        """One step of optimization based on rollout memory"""
        with torch.no_grad():
            self.preprocess_rollout()
        
        # DEEP-RL TUTORIALS: КОСТЫЛЬ
        #self.advantages = self.returns[:-1] - self.values[:-1]
        #self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)
        
        # going through rollout several (config.epochs) times:
        for epoch in range(self.config.epochs):
            # TODO: drop last = False? What if there is 1 sample?
            sampler = BatchSampler(SubsetRandomSampler(range(self.env.num_envs * self.config.rollout)), self.config.batch_size, drop_last=False)
            
            for indices in sampler:
                # retrieving new batch as part of rollout
                self.returns_b = self.returns.view(-1, *self.config.value_repr_shape)[indices]
                self.old_values_b = self.values.view(-1, *self.config.value_repr_shape)[indices]
                self.old_action_log_probs_b = self.action_log_probs.view(-1)[indices]
                #self.advantages_b = self.advantages.view(-1)[indices]  # КОСТЫЛЬ
                
                # calculating current value, action_log_prob, entropy
                dist, self.values_b = self.policy(self.observations.view(-1, *self.config.observation_shape)[indices])
                self.values_b = self.values_b.squeeze()  # IMPORTANT ([32] - [32, 1] problem)
                self.action_log_probs_b = dist.log_prob(self.actions.view(-1, *self.config.actions_shape)[indices])#.sum(dim=-1)        
                self.entropy_b = dist.entropy()#.sum(dim=-1)
                
                # performing step
                self.gradient_ascent_step()
  return PPO
