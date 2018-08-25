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
    
    def __init__(self, replay_buffer_nsteps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.replay_buffer_nsteps = replay_buffer_nsteps
        self.nstep_buffer = []
    
    def memorize(self, state, action, reward, next_state, done):        
        self.nstep_buffer.append((state, action, reward, next_state))
        
        if len(self.nstep_buffer) >= self.replay_buffer_nsteps:      
            nstep_reward = sum([self.nstep_buffer[i][2] * (self.gamma**i) for i in range(self.replay_buffer_nsteps)])
            state, action, _, _ = self.nstep_buffer.pop(0)
            super().memorize(state, action, nstep_reward, next_state, done)
            
        if done:
            self.nstep_buffer = []
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.nstep_buffer, mem_f)
        pickle.dump(self.replay_buffer_nsteps, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.nstep_buffer = pickle.load(mem_f)
        self.replay_buffer_nsteps = pickle.load(mem_f)
  return NstepReplay
  
#?            
'''
class CollectiveNstepReplayBufferAgent(NstepReplayBufferAgent):
    """
    Experimental. Stores all transitions from transitions on one step to transitions on n steps.
    """
    __doc__ += NstepReplayBufferAgent.__doc__
        
    def memorize(self, state, action, reward, next_state, done):
        self.nstep_buffer.append((state, action, reward, next_state))
        
        R = 0
        for i in range(len(self.nstep_buffer) - 1, -1, -1):
            R *= self.gamma
            R += self.nstep_buffer[i][2] * self.gamma
            ReplayBufferAgent.memorize(self, self.nstep_buffer[i][0], self.nstep_buffer[i][1], R, next_state, done)           
        
        if len(self.nstep_buffer) >= self.replay_buffer_nsteps:      
            self.nstep_buffer.pop(0)
            
        if done:
            self.nstep_buffer = []
'''
