from .DQN import *

class DDPG_Actor(Head):
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.head = nn.Sequential(
            self.linear(self.feature_size, config.num_actions),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.head(self.feature_extractor_net(state)).view(-1, *self.config.actions_shape)

class DDPG_Critic(QnetworkHead):
    def __init__(self, config, name):
        super().__init__(config, name)
        
        self.head = self.linear(self.feature_size, 1)
        
        # avoiding actor being part of this nn.Module
        actor = self.config.ActorHead(self.config, "Actor").to(device)
        actor.init_optimizer()
        self.actor = lambda: actor
        
    def get_feature_size(self):
        return self.feature_extractor_net(Tensor(size=(10, *self.config["observation_shape"])),
                                          Tensor(size=(10, *self.config["actions_shape"]))).size()[1]
        
    def forward(self, state):
        return state
    
    def greedy(self, output):
        return self.actor()(output)
        
    def gather(self, output, action_b):
        return self.head(self.feature_extractor_net(output, action_b)).squeeze(dim=1)
    
    def value(self, output):
        return self.head(self.feature_extractor_net(output, self.greedy(output))).squeeze(dim=1)

def DDPG_QAgent(parclass):
  """Requires parent class, inherited from Agent.
  Already inherits from QAgent"""
    
  class DDPG_QAgent(QAgent(parclass)):
    """
    Deep Deterministic Policy Gradient
    Based on: https://arxiv.org/abs/1509.02971
    
    Args:
        ActorHead - class of deterministic actor head, inherited from DDPG_Actor
    """
    __doc__ += QAgent(parclass).__doc__
    PARAMS = QAgent(parclass).PARAMS | Head.PARAMS("Actor") 
    
    def __init__(self, config):
        config.setdefault("QnetworkHead", DDPG_Critic)        
        assert issubclass(config["QnetworkHead"], DDPG_Critic), "DDPG requires QnetworkHead to be inherited from DDPG_Head class"  
        
        self.config.setdefault("ActorHead", DDPG_Actor)
        assert issubclass(self.config["ActorHead"], DDPG_Actor), "DDPG requires ActorHead to be inherited from DDPG_Actor class"        
        
        super().__init__(config)
    
    def act(self, state, record=False):
        self.q_net.actor().eval()
        return super().act(state, record)
    
    def optimize_model(self, q_network):
        q_network.actor().train()
        
        # Basic QAgent just optimizes critic!
        super().optimize_model(q_network)
        
        # Now we need actor optimization.
        state_b, weights_b = self.batch[0], self.batch[-1]
        
        loss_b = -q_network.value(state_b)        
        loss = (loss_b * weights_b).mean()
        self.logger["actor_loss"].append(loss.detach().cpu().numpy())
        
        q_network.actor().optimize(loss)       
            
    def show_record(self):
        show_frames(self.record["frames"])
        
  return DDPG_QAgent
