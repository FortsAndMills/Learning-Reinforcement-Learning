from .network_heads import *
from .utils import *

class InverseModelHead(Head):
    pass

class InverseModelNetwork(InverseModelHead):
    def __init__(self, config, name):
        super().__init__(config, name)

        self.head = self.linear(self.feature_size * 2, config.num_actions)

    def forward(self, state, next_state):
        f1 = self.feature_extractor_net(state)
        f2 = self.feature_extractor_net(next_state)
        return self.head(torch.cat([f1, f2], dim=1))

def InverseModel(parclass):
  """Requires parent class, inherited from Agent."""
    
  class InverseModel(parclass):
    """
    Self-supervision based intrinsic motivation generation
    Based on: https://arxiv.org/abs/1705.05363
    
    Args:
        InverseModelHead - class of InverseModelHead, inherited from InverseModelHead
        curiosity_batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching curiosity optimization, int
        curiosity_optimize_iterations - number of gradient descent steps after one transition, can be fractional, float
        curiosity_coeff - coeff to multiply on intrinsic reward, float
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | Head.PARAMS("InverseModel") | {"InverseModelHead", "curiosity_batch_size", 
                                                          "replay_buffer_init", "curiosity_optimize_iterations"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("InverseModelHead", InverseModelNetwork)
        self.config.setdefault("curiosity_batch_size", 32)
        self.config.setdefault("replay_buffer_init", 1000)
        self.config.setdefault("curiosity_optimize_iterations", 1)
        self.config.setdefault("curiosity_coeff", 1)
        
        assert self.config.replay_buffer_init >= self.config.batch_size, "Batch size must be smaller than replay_buffer_init!"
        
        self.inverse_model = self.config.InverseModelHead(self.config, "InverseModel").to(device)
        self.inverse_model.init_optimizer()
        self.curiosity_optimize_iteration_charges = 0

        self.logger_labels["full_rewards"] = ("episode", "reward")

    def action_prediction_loss(self, action, predicted_action):
        """
        Get loss for predicted action:
        input: action - true action, Tensor, ints or floats, (batch_size x actions_shape)
        input: predicted_action - Tensor, ints or floats, (batch_size x actions_shape)
        output: loss - Tensor, float, (batch_size) 
        """
        return torch.nn.CrossEntropyLoss(reduction="none")(predicted_action, action)

    def intrinsic_motivation(self, state, action, next_state, done):
        """
        Get intrinsic reward for current transitions:
        input: state - numpy array, (batch_size x observation_shape)
        input: action - numpy array, ints or floats, (batch_size x actions_shape)
        input: next_state - numpy array, (batch_size x observation_shape)
        input: done - numpy array, 0 and 1, (batch_size)
        output: intrinsic_reward - numpy array, floats, (batch_size)
        """
        # TODO what about done=True?
        self.inverse_model.eval()
        with torch.no_grad():
            return (self.config.curiosity_coeff * 
                self.action_prediction_loss(self.ActionTensor(action), self.inverse_model(Tensor(state), Tensor(next_state))).cpu().numpy())
    
    def see(self, state, action, reward, next_state, done):
        # intrinsic motivation
        intrinsic_reward = self.intrinsic_motivation(state, action, next_state, done)
        super().see(state, action, reward + intrinsic_reward, next_state, done)

        # logging full reward
        self.full_R += reward + intrinsic_reward
        for res in self.full_R[done]:
            self.logger["full_rewards"].append(res)            
        self.full_R[done] = 0
        
        # performing optimization of curiosity networks
        self.curiosity_optimize_iteration_charges += self.config.curiosity_optimize_iterations * self.env.num_envs
        while self.curiosity_optimize_iteration_charges >= 1:
            self.curiosity_optimize_iteration_charges -= 1
            
            if len(self) >= self.config.replay_buffer_init:
                self.optimize_curiosity()
    
    def reset(self):
        super().reset()
        self.full_R = np.zeros((self.env.num_envs), dtype=np.float32)

    def optimize_curiosity(self):
        '''
        One step of curiosity network optimization:
        '''        
        self.batch = self.sample(self.config.batch_size)
        state_b, action_b, reward_b, next_state_b, done_b, weights_b = self.batch

        # getting loss
        self.inverse_model.train()
        loss_b = self.action_prediction_loss(action_b, self.inverse_model(state_b, next_state_b))
        assert len(loss_b.shape) == 1, loss_b
        
        self.inverse_model.optimize(loss_b.mean())
        
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.inverse_model.load_state_dict(torch.load(name + "-inversemodel"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.inverse_model.state_dict(), name + "-inversemodel")
  return InverseModel






