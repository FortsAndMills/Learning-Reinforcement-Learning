from .replayBuffer import *

def NstepReplay(parclass):
  """Requires parclass inherited from ReplayBufferAgent"""
  
  class NstepReplay(parclass):
    """
    Stores transitions more than on one step.
    
    Args:
        replay_buffer_nsteps - N steps, int
    """
    __doc__ += parclass.__doc__
    PARAMS = parclass.PARAMS | {"replay_buffer_nsteps"}
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("replay_buffer_nsteps", 3)
        self.nstep_buffer = []
    
    def memorize(self, state, action, reward, next_state, done):        
        self.nstep_buffer.append((state, action, reward, next_state, np.ones(action.shape, dtype=bool)))
        
        if len(self.nstep_buffer) >= self.config.replay_buffer_nsteps:      
            nstep_reward = sum([self.nstep_buffer[i][2] * (self.config.gamma**i) for i in range(self.config.replay_buffer_nsteps)])
            state, action, _, _, actual = self.nstep_buffer.pop(0)
            super().memorize(state[actual], action[actual], nstep_reward[actual], next_state[actual], done[actual])
            
        for i in range(len(self.nstep_buffer)):
            self.nstep_buffer[i][-1][done] = False
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.nstep_buffer, mem_f)
        pickle.dump(self.replay_buffer_nsteps, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.nstep_buffer = pickle.load(mem_f)
        self.replay_buffer_nsteps = pickle.load(mem_f)
  return NstepReplay
  

def CollectiveNstepReplayBufferAgent(parclass):
  """
  Requires parclass inherited from ReplayBufferAgent.
  Already inherits from NstepReplay
  """
  
  class CollectiveNstepReplayBufferAgent(NstepReplay(parclass)):
    """
    Experimental. Stores all transitions from transitions on one step to transitions on n steps.
    """
    __doc__ += NstepReplay(parclass).__doc__
        
    def memorize(self, state, action, reward, next_state, done):
        self.nstep_buffer.append((state, action, reward, next_state, np.ones(action.shape, dtype=bool)))
        
        R = np.zeros((state.shape[0]))
        for i in reversed(range(len(self.nstep_buffer))):
            R *= self.gamma
            R += self.nstep_buffer[i][2] * self.gamma
            actual = self.nstep_buffer[i][-1]
            ReplayBufferAgent.memorize(self, self.nstep_buffer[i][0][actual], self.nstep_buffer[i][1][actual], R[actual], next_state[actual], done[actual])           
        
        if len(self.nstep_buffer) >= self.config.replay_buffer_nsteps:      
            self.nstep_buffer.pop(0)
            
        for i in range(len(self.nstep_buffer)):
            self.nstep_buffer[i][-1][done] = False
  return CollectiveNstepReplayBufferAgent

