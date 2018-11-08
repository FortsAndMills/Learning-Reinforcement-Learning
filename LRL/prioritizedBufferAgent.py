from .replayBuffer import *

class PrioritizedBufferAgent(ReplayBufferAgent):
    """
    Prioritized replay memory based on weighted importance sampling.
    Proxy of priority is considered to be loss on given transition. For DQN it is absolute of td loss.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        rp_alpha - degree of prioritezation, float, from 0 to 1
        rp_beta_start - degree of importance sampling smoothing out the bias, float, from 0 to 1
        rp_beta_frames - number of frames till unbiased sampling
        clip_priorities - clipping priorities as suggested in original paper
    """
    __doc__ += ReplayBufferAgent.__doc__ 
      
    def __init__(self, config):
        super().__init__(config)
        
        self.priorities = np.array([])
        self.clip_priorities = config.get("clip_priorities", 1)
               
        self.rp_alpha = config.get("rp_alpha", 0.6) 
        self.rp_beta_by_frame = lambda frame_idx: min(1.0, config.get("rp_beta_start", 0.4) + frame_idx * (1.0 - config.get("rp_beta_start", 0.4)) / config.get("rp_beta_frames", 100000))

    def memorize_transition(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self) < self.replay_buffer_capacity:
            self.priorities = np.append(self.priorities, max_priority)
        else:
            self.priorities[self.pos] = max_priority
            
        super().memorize_transition(state, action, reward, next_state, done)

    def sample(self, batch_size):
        probs  = np.array(self.priorities) ** self.rp_alpha
        probs /= probs.sum()
        
        self.batch_indices = np.random.choice(len(self), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in self.batch_indices]
               
        weights  = (probs[self.batch_indices]) ** (-self.rp_beta_by_frame(self.frames_done))
        weights /= (probs.min()) ** (-self.rp_beta_by_frame(self.frames_done))
       
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, weights

    def update_priorities(self, batch_priorities):
        self.priorities[self.batch_indices] = (batch_priorities + 1e-5).clip(max=self.clip_priorities)
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.priorities, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.priorities = pickle.load(mem_f)
