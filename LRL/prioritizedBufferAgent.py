from .replayBuffer import *

class PrioritizedBufferAgent(ReplayBufferAgent):  # TODO check if sumtree works faster
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
    PARAMS = ReplayBufferAgent.PARAMS | {"rp_alpha", "rp_beta_start", "rp_beta_frames", "clip_priorities"}
      
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("clip_priorities", 1)
        self.config.setdefault("rp_alpha", 0.6)
        self.config.setdefault("rp_beta_start", 0.4)
        self.config.setdefault("rp_beta_frames", 100000)
        
        self.priorities = np.array([])
        self.max_priority = 1.0
                
        self.rp_beta_by_frame = lambda frame_idx: min(1.0, 
                self.config.rp_beta_start + frame_idx * (1.0 - self.config.rp_beta_start) / self.config.rp_beta_frames)
                
        self.logger_labels["median weight"] = ("training iteration", "median weight")

    def memorize_transition(self, state, action, reward, next_state, done):
        # new transition is stored with max priority
        if len(self) < self.config.replay_buffer_capacity:
            self.priorities = np.append(self.priorities, self.max_priority)
        else:
            self.priorities[self.pos] = self.max_priority
            
        super().memorize_transition(state, action, reward, next_state, done)

    def sample(self, batch_size):
        # proposed weights for sampling
        probs  = np.array(self.priorities) ** self.config.rp_alpha
        probs /= probs.sum()
        
        self.batch_indices = np.random.choice(len(self), batch_size, p=probs, replace=True)   # do not use replace = False, it makes O(n)
        samples = [self.buffer[idx] for idx in self.batch_indices] # seems like the fastest code for sampling!
        
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        weights  = (probs[self.batch_indices]) ** (-self.rp_beta_by_frame(self.frames_done))
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        weights /= (probs.min()) ** (-self.rp_beta_by_frame(self.frames_done))
       
        states, actions, rewards, next_states, dones = zip(*samples)
        self.logger["median weight"].append(np.median(weights))
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, weights

    def update_priorities(self, batch_priorities):
        """
        Updates priorities for previously sampled batch, indexes stored in self.batch_indices
        input: batch_priorities - np.array, floats, (batch_size)
        """
        self.priorities[self.batch_indices] = (batch_priorities + 1e-5).clip(max=self.config.clip_priorities)
        self.max_priority = max(self.max_priority, self.priorities[self.batch_indices].max())
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.priorities, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.priorities = pickle.load(mem_f)
