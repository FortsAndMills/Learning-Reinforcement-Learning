from .utils import *
from .network_heads import *

def QAgent(parclass):
  """Requires parent class, inherited from Agent."""
    
  class QAgent(parclass):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        FeatureExtractorNet - class inherited from nn.Module
        QnetworkHead - class of Q-network head, inherited from QnetworkHead
        linear_layer - default linear layer, nn.Linear or NoisyLinear
        gamma - infinite horizon protection, float, from 0 to 1
        batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching q-network optimization
        optimize_iterations - number of gradient descent steps after one transition, int
        optimizer - class inherited from torch.optimizer, Adam by default
        optimizer_args - arguments for optimizer, dictionary
        grad_norm_max - max norm of gradients for clipping, float
    """
    __doc__ += parclass.__doc__
    
    def __init__(self, config):
        super().__init__(config)
        
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.replay_buffer_init = config.get("replay_buffer_init", 1000)
        self.optimize_iterations = config.get("optimize_iterations", 1)
        self.grad_norm_max = config.get("grad_norm_max", 1)
        
        self.policy_net = self.init_network() 
        self.optimizer = config.get("optimizer", optim.Adam)(self.policy_net.parameters(), **config.get("optimizer_args", {}))
        
        self.logger_labels["loss"] = ("training iteration", "loss")
        if self.config.get("linear_layer", nn.Linear) is NoisyLinear:
            self.logger_labels["magnitude"] = ("training game step", "noise magnitude")

    def init_network(self):
        '''create a new Q-network'''
        net = self.config.get("QnetworkHead", Qnetwork)(self.config)        
        net.after_init()
        return net
    
    def act(self, state):
        if self.learn:
            self.policy_net.train()
        else:
            self.policy_net.eval()
        
        with torch.no_grad():
            self.qualities = self.policy_net(Tensor(state))
            return self.policy_net.greedy(self.qualities).cpu().numpy()

    def see(self, state, action, reward, next_state, done):
        super().see(state, action, reward, next_state, done)
        
        for i in range(self.optimize_iterations):
            self.optimize_model()
        
        if self.config.get("linear_layer", nn.Linear) is NoisyLinear:
            self.logger["magnitude"].append(self.policy_net.magnitude())
       
    def estimate_next_state(self, next_state_b):
        '''
        Calculates estimation of next state
        input: next_state_batch - FloatTensor, batch_size x state_dim
        output: FloatTensor, batch_size
        '''
        return self.policy_net.value(self.policy_net(next_state_b))
    
    def batch_target(self, reward_b, next_state_b, done_b):
        '''
        Calculates target for batch to learn
        input: reward_batch - FloatTensor, batch_size
        input: next_state_batch - FloatTensor, batch_size x state_dim
        input: done_batch - FloatTensor, batch_size
        output: FloatTensor, batch_size
        '''
        next_q_values = self.estimate_next_state(next_state_b)
        return reward_b + (self.gamma**self.replay_buffer_nsteps) * next_q_values * (1 - done_b)

    def get_loss(self, y, guess):
        '''
        Calculates batch loss
        input: y - target, FloatTensor, batch_size
        input: guess - current model output, FloatTensor, batch_size
        output: FloatTensor, batch_size
        '''
        return (guess - y).pow(2)
        
    def get_transition_importance(self, loss_b):
        '''
        Calculates importance of transitions in batch by loss
        input: loss_b - FloatTensor, batch_size
        output: FloatTensor, batch_size
        '''
        return loss_b**0.5

    def optimize_model(self):
        '''One step of Q-network optimization'''
        if len(self) < self.replay_buffer_init:
            return
        
        state_b, action_b, reward_b, next_state_b, done_b, weights_b = self.sample(self.batch_size)

        state_b      = Tensor(np.float32(state_b))
        next_state_b = Tensor(np.float32(next_state_b))
        action_b     = LongTensor(action_b)
        reward_b     = Tensor(reward_b)
        done_b       = Tensor(done_b)
        weights_b    = Tensor(weights_b)
        
        self.policy_net.train()
        q_values      = self.policy_net.gather(self.policy_net(state_b), action_b)
        with torch.no_grad():
            target_q_values = self.batch_target(reward_b, next_state_b, done_b)

        loss_b = self.get_loss(target_q_values, q_values)
        self.update_priorities(self.get_transition_importance(loss_b).detach().cpu().numpy())
        
        loss = (loss_b * weights_b).mean()
        self.logger["loss"].append(loss.detach().cpu().numpy())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm_max)
        self.optimizer.step()
    
    def record_step(self):
        super().record_step()
        self.record["qualities"].append(self.qualities.cpu().numpy())
        
    def show_record(self):
        super().show_record()  #TODO: show near the game plot of action qualities
    
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.policy_net.load_state_dict(torch.load(name + "-qnet"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.policy_net.state_dict(), name + "-qnet")
  return QAgent
