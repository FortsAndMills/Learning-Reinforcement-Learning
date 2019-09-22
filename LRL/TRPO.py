from .A2C import *
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

# DOES NOT WORK :(

def TRPO(parclass):
  """Requires parent class, inherited from A2C."""
    
  class TRPO(parclass):
    """
    Trust-Region Policy Optimization algorithm (TRPO).
    Requires parent class inherited from A2C.
    Based on: https://arxiv.org/abs/1502.05477
    
    Args:
        max_kl - constraint limitation
        cg_iters - number of iterations of conjugate gradients algorithm
        residual_tol - tolerance to earlier break for conjugate gradients algorithm
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | {"max_kl", "cg_iters", "residual_tol"} 
    
    def __init__(self, config):
        super().__init__(config)        
        self.config.setdefault("rollout", 1024)
        self.config.setdefault("cg_iters", 10)
        self.config.setdefault("max_kl", 0.001)
        self.config.setdefault("residual_tol", 1e-7)
        
        self.logger_labels["kl_change"] = ("training iteration", "KL between new policy and old one")
        self.logger_labels["acceptance_ratio"] = ("training iteration", "acceptance_ratio")
        self.logger_labels["expected_improvement"] = ("training iteration", "expected_improvement")
    
    def optimized_function(self):
        advantages = self.returns_b - self.values_b
        return -(advantages.detach() * (self.new_action_log_probs_b - self.action_log_probs_b.detach()).exp())
        
    def critic_loss(self):
        return (self.returns_b.detach() - self.new_values_b.view(-1)).pow(2)
        
    def entropy_loss(self):
        return -self.new_entropy_b
    
    def surrogate_function(self, write_to_log=False):
        # evaluate new policy
        self.new_action_dist, self.new_values = self.policy(self.observations.view(-1, *self.config.observation_shape))
        self.new_action_log_probs = self.new_action_dist.log_prob(self.actions.view(-1, *self.config.actions_shape))#.sum(dim=-1)
        
        # constructing current batch
        self.new_values = self.new_values.view(self.config.rollout + 1, self.env.num_envs, *self.config.value_repr_shape)
        self.new_action_log_probs = self.new_action_log_probs.view(self.config.rollout + 1, self.env.num_envs)
                
        self.new_values_b = self.new_values[:-1].view(-1)
        self.new_action_log_probs_b = self.new_action_log_probs[:-1].view(-1)    
        self.new_entropy_b = self.action_dist.entropy()[:-1].view(-1)
        
        # calculating loss        
        actor_loss = self.optimized_function().mean()
        critic_loss = self.critic_loss().mean()        
        entropy_loss = self.entropy_loss().mean()
        
        loss = actor_loss + self.config.critic_loss_weight * critic_loss + self.config.entropy_loss_weight * entropy_loss
        
        # logging
        if write_to_log:
            self.logger["actor_loss"].append(actor_loss.item())
            self.logger["critic_loss"].append(critic_loss.item())
            self.logger["entropy_loss"].append(entropy_loss.item())
            
        return loss
    
    def mean_kl_divergence(self):
        """returns an estimate of the average KL divergence between a model used to collect roll-out and self.policy"""
        kl_policy = torch.sum(self.action_dist.probs.detach() * (self.action_dist.logits.detach() - self.new_action_dist.logits), dim=1)
        assert (kl_policy > -0.001).all(), "WTF?"
        
        kl_policy = kl_policy.mean()
        kl_values = ((self.values.detach() - self.new_values)**2).mean() / 2
        
        return kl_policy + self.config.critic_loss_weight * kl_values
        
    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        """
        self.policy.optimizer.zero_grad()
        mean_kl_div = self.mean_kl_divergence()
        kl_grad = torch.autograd.grad(mean_kl_div, self.policy.parameters(), create_graph=True, retain_graph=True)
        
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        grad_grad = torch.autograd.grad(grad_vector_product, self.policy.parameters(), retain_graph=True)
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grad]).data
        
        return fisher_vector_product# + (self.cg_damping * vector.data)
    
    def conjugate_gradient(self, b):
        """
        Returns F^(-1)b where F is the Hessian of the KL divergence
        input: b - 1D numpy vector
        output: x - 1D numpy vector
        """
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)
        for _ in range(self.config.cg_iters):
          z = self.hessian_vector_product(Tensor(p)).squeeze(0).cpu().numpy()
          v = rdotr / p.dot(z)
          x += v * p
          r -= v * z
          newrdotr = r.dot(r)
          mu = newrdotr / rdotr
          p = r + mu * p
          rdotr = newrdotr
          if rdotr < self.config.residual_tol:
            break
        return x
        
    def linesearch(self, x, fullstep, expected_improve_rate):
        """
        Returns the parameter vector given by a linesearch
        input: x - Tensor, 1D, current parameters
        input: fullstep - Tensor, 1D, direction (natural gradient), normalized
        input: expected_improve_rate - ?!?
        output: new parameters - Tensor, 1D
        """
        accept_ratio = 0.1
        max_backtracks = 10
        with torch.no_grad():
            fval = self.surrogate_function().mean()
            
        for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
            xnew = x.data.cpu().numpy() + stepfrac * fullstep
            vector_to_parameters(Tensor(xnew), self.policy.parameters())
            with torch.no_grad():
                newfval = self.surrogate_function().mean()
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            #print(actual_improve, " ", expected_improve)
            if ratio > accept_ratio and actual_improve > 0:
                #print("Accepted")
                self.logger["acceptance_ratio"].append(ratio)
                self.logger["expected_improvement"].append(expected_improve_rate)
                return Tensor(xnew)
        
        raise Exception("Line search error")
        return x   
    
    def gradient_ascent_step(self):
        """Makes one update of policy weights"""
        
        # get loss
        loss = self.surrogate_function(write_to_log=True)
        
        # calculating gradient
        self.policy.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        policy_gradient = parameters_to_vector([v.grad for v in self.policy.parameters()]).squeeze(0)        
        assert policy_gradient.nonzero().size()[0] > 0, "Policy gradient is 0. Skipping update?.."
        
        # Use conjugate gradient algorithm to determine the step direction in theta space
        step_direction = self.conjugate_gradient(-policy_gradient.cpu().numpy())

        # Do line search to determine the stepsize of theta in the direction of step_direction
        shs = step_direction.dot(self.hessian_vector_product(Tensor(step_direction)).cpu().numpy().T) / 2
        lm = np.sqrt(shs / self.config.max_kl)
        fullstep = step_direction / lm
        gdotstepdir = -policy_gradient.dot(Tensor(step_direction)).data[0]
        theta = self.linesearch(parameters_to_vector(self.policy.parameters()), fullstep, gdotstepdir / lm)

        # Update parameters of policy model
        if any(np.isnan(theta.data.cpu().numpy())):
          raise Exception("NaN detected. Skipping update...")
        else:
          vector_to_parameters(theta, self.policy.parameters())

        kl_old_new = self.mean_kl_divergence()
        self.logger["kl_change"].append(kl_old_new.item())
        
  return TRPO
