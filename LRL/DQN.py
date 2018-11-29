from .utils import *
from .network_heads import *

def QAgent(parclass):
  """Requires parent class, inherited from Agent."""
    
  class QAgent(parclass):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        QnetworkHead - class of Q-network head, inherited from QnetworkHead
        gamma - infinite horizon protection, float, from 0 to 1
        batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching q-network optimization
        optimize_iterations - number of gradient descent steps after one transition, can be fractional, float
        optimizer - class inherited from torch.optimizer, Adam by default
        optimizer_args - arguments for optimizer, dictionary
        grad_norm_max - max norm of gradients for clipping, float
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("Qnetwork") | {"QnetworkHead", "gamma", "batch_size", 
                                                          "replay_buffer_init", "optimize_iterations"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("QnetworkHead", Qnetwork)
        self.config.setdefault("gamma", 0.99)
        self.config.setdefault("batch_size", 32)
        self.config.setdefault("replay_buffer_init", 1000)
        self.config.setdefault("optimize_iterations", 1)
        
        assert self.config.replay_buffer_init >= self.config.batch_size, "Batch size must be smaller than replay_buffer_init!"
        assert self.config.gamma > 0 and self.config.gamma <= 1, "Gamma must lie in (0, 1]"
        
        self.q_net = self.config.QnetworkHead(self.config, "Qnetwork").to(device)
        self.q_net.init_optimizer()
        self.optimize_iteration_charges = 0
        
        self.logger_labels["loss"] = ("training iteration", "loss")
        self.log_net(self.q_net, "Qnetwork")
    
    def act(self, state, record=False):
        self.q_net.eval()
        
        with torch.no_grad():
            qualities = self.q_net(Tensor(state))
            
            if record:
                self.record["qualities"].append(qualities[0:1].cpu().numpy())
            
            return self.q_net.greedy(qualities).cpu().numpy()

    def see(self, state, action, reward, next_state, done):
        super().see(state, action, reward, next_state, done)
        
        self.optimize_iteration_charges += self.config.optimize_iterations * self.env.num_envs
        while self.optimize_iteration_charges >= 1:
            self.optimize_iteration_charges -= 1
            
            if len(self) >= self.config.replay_buffer_init:
                self.optimize_model()
       
    def estimate_next_state(self, next_state_b):
        '''
        Calculates estimation of next state.
        May use self.next_state_q, which is an output of self.q_net on next_state_b
        input: next_state_batch - FloatTensor, (batch_size x state_dim)
        output: FloatTensor, batch_size
        '''
        return self.q_net.value(self.next_state_q)
    
    def batch_target(self, reward_b, next_state_b, done_b):
        '''
        Calculates target for batch to learn
        input: reward_batch - FloatTensor, (batch_size)
        input: next_state_batch - FloatTensor, (batch_size x state_dim)
        input: done_batch - FloatTensor, (batch_size)
        output: FloatTensor, (batch_size)
        '''
        next_q_values = self.estimate_next_state(next_state_b)
        return reward_b + (self.config.gamma**self.config.replay_buffer_nsteps) * next_q_values * (1 - done_b)

    def get_loss(self, y, guess):
        '''
        Calculates batch loss
        input: y - target, FloatTensor, (batch_size)
        input: guess - current model output, FloatTensor, (batch_size)
        output: FloatTensor, (batch_size)
        '''
        return (guess - y).pow(2)
        
    def get_transition_importance(self, loss_b):
        '''
        Calculates importance of transitions in batch by loss
        input: loss_b - FloatTensor, (batch_size)
        output: FloatTensor, (batch_size)
        '''
        return loss_b**0.5

    def optimize_model(self):
        '''One step of Q-network optimization'''        
        self.batch = self.sample(self.config.batch_size)
        state_b, action_b, reward_b, next_state_b, done_b, weights_b = self.batch
        # TODO: weights logging?

        state_b      = Tensor(np.float32(state_b))
        next_state_b = Tensor(np.float32(next_state_b))
        action_b     = self.ActionTensor(action_b)
        reward_b     = Tensor(np.float32(reward_b))
        done_b       = Tensor(np.float32(done_b))
        weights_b    = Tensor(weights_b)
        
        # optimizing forward pass through net!
        self.q_net.train()
        output = self.q_net(torch.cat([state_b, next_state_b], dim=0))
        self.state_q, self.next_state_q = torch.split(output, self.config.batch_size, dim=0)
        
        # getting q values for state and next state
        q_values = self.q_net.gather(self.state_q, action_b)
        with torch.no_grad():
            target_q_values = self.batch_target(reward_b, next_state_b, done_b)
            
        # getting loss and updating transition importances
        loss_b = self.get_loss(target_q_values, q_values)
        self.update_priorities(self.get_transition_importance(loss_b).detach().cpu().numpy())
        
        assert len(loss_b.shape) == 1, loss_b
        
        # making optimization step
        loss = (loss_b * weights_b).mean()
        self.logger["loss"].append(loss.detach().cpu().numpy())
        
        self.q_net.optimize(loss)
        
    def show_record(self):
        show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"]), "Qualities", np.arange(self.config["num_actions"]))
    
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.q_net.load_state_dict(torch.load(name + "-qnet"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.q_net.state_dict(), name + "-qnet")
  return QAgent
