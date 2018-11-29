from .DQN import *

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
        super().__init__(config)
        
        self.config.setdefault("ActorHead", DDPG_Actor)
        assert issubclass(self.config["ActorHead"], DDPG_Actor), "DDPG requires ActorHead to be inherited from DDPG_Actor class"         
        
        self.actor = self.config.ActorHead(self.config, "Actor").to(device)
        self.actor.init_optimizer()
        self.config["actor"] = self.actor
                
        self.logger_labels["actor_loss"] = ("training iteration", "actor_loss")
        self.log_net(self.actor, "Actor")
    
    def act(self, state, record=False):
        self.actor.eval()
        return super().act(state, record)
    
    def optimize_model(self):
        self.actor.train()
        
        # Basic QAgent just optimizes critic!
        super().optimize_model()
        
        # Now we need actor optimization. We need only batch of states for that.
        state_b = Tensor(np.float32(self.batch[0]))
        weights_b = Tensor(self.batch[-1])
        
        loss_b = -self.q_net.value(state_b)        
        loss = (loss_b * weights_b).mean()
        self.logger["actor_loss"].append(loss.detach().cpu().numpy())
        
        self.actor.optimize(loss)       
            
    def show_record(self):
        show_frames(self.record["frames"])
        
  return DDPG_QAgent
