from .utils import *
from .network_heads import *

class QnetworkHead(Head):
    def greedy(self, output):
        '''returns greedy action based on output of net'''
        return output.max(1)[1]
        
    def gather(self, output, action_b):
        '''returns output of net for given batch of actions'''
        return output.gather(1, action_b.unsqueeze(1)).squeeze(1)
    
    def value(self, output):
        '''returns output for greedily-chosen action'''
        return output.max(1)[0]

class Qnetwork(QnetworkHead):
    '''Simple Q-network head'''
    def __init__(self, config, name): 
        super().__init__(config, name)      
        self.head = self.linear(self.feature_size, config.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        return self.head(state)
        
class DuelingQnetwork(QnetworkHead):
    '''Dueling version of Q-network head'''
    def __init__(self, config, name): 
        super().__init__(config, name)       
        self.v_head = self.linear(self.feature_size, 1)
        self.a_head = self.linear(self.feature_size, config.num_actions)
        
    def forward(self, state):
        state = self.feature_extractor_net(state)
        v = self.v_head(state)
        a = self.a_head(state)
        return v + a - a.mean(dim=1, keepdim=True)

def QAgent(parclass):
  """Requires parent class, inherited from Agent."""
    
  class QAgent(parclass):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        QnetworkHead - class of Q-network head, inherited from QnetworkHead
        batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching q-network optimization
        optimize_iterations - number of gradient descent steps after one transition, can be fractional, float
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("Qnetwork") | {"QnetworkHead", "gamma", "batch_size", 
                                                          "replay_buffer_init", "optimize_iterations"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("QnetworkHead", Qnetwork)
        self.config.setdefault("batch_size", 32)
        self.config.setdefault("replay_buffer_init", 1000)
        self.config.setdefault("optimize_iterations", 1)
        
        assert self.config.replay_buffer_init >= self.config.batch_size, "Batch size must be smaller than replay_buffer_init!"
        assert self.config.gamma > 0 and self.config.gamma <= 1, "Gamma must lie in (0, 1]"
        
        self.q_net = self.config.QnetworkHead(self.config, "Qnetwork").to(device)
        self.q_net.init_optimizer()
        self.optimize_iteration_charges = 0
    
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
                self.optimize_model(self.q_net)
       
    def estimate_next_state(self, next_state_b):
        '''
        Calculates estimation of next state.
        input: next_state_batch - FloatTensor, (batch_size x state_dim)
        output: FloatTensor, batch_size
        '''
        return self.q_net.value(self.q_net(next_state_b))
    
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

    def optimize_model(self, q_network):
        '''
        One step of Q-network optimization:
        input: network - instance of nn.Module
        '''        
        self.batch = self.sample(self.config.batch_size)
        state_b, action_b, reward_b, next_state_b, done_b, weights_b = self.batch

        state_b      = Tensor(state_b)
        next_state_b = Tensor(next_state_b)
        action_b     = self.ActionTensor(action_b)
        reward_b     = Tensor(reward_b)
        done_b       = Tensor(done_b)
        weights_b    = Tensor(weights_b)
        
        q_network.train()
        
        # getting q values for state and next state
        q_values = q_network.gather(q_network(state_b), action_b)
        with torch.no_grad():
            target_q_values = self.batch_target(reward_b, next_state_b, done_b)
            
        # getting loss
        loss_b = self.get_loss(target_q_values, q_values)        
        assert len(loss_b.shape) == 1, loss_b
        
        # updating transition importances
        self.update_priorities(self.get_transition_importance(loss_b).detach().cpu().numpy())
        
        # making optimization step
        loss = (loss_b * weights_b).mean()
        
        q_network.optimize(loss)
        
    def show_record(self):
        show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"]), "Qualities", np.arange(self.config["num_actions"]))
    
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.q_net.load_state_dict(torch.load(name + "-qnet"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.q_net.state_dict(), name + "-qnet")
  return QAgent
