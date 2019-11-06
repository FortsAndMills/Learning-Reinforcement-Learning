from .replayBuffer import *

class MyBufferAgent(ReplayBufferAgent):  # TODO check if sumtree works faster
    """
    Experimental
    """
    __doc__ += ReplayBufferAgent.__doc__ 
      
    def __init__(self, config):
        super().__init__(config)
        
        self.true_reward_to_go = np.array([])
        self.priorities = np.array([])
        self.logger_labels["median weight"] = ("training iteration", "median weight")

    def memorize_transition(self, state, action, reward, next_state, done):
        # new transition is stored with max priority
        if len(self) < self.config.replay_buffer_capacity:
            self.priorities = np.append(self.priorities, 1)
            self.true_reward_to_go = np.append(self.true_reward_to_go, reward)
        else:
            self.priorities[self.pos] = 1
            self.true_reward_to_go[self.pos] = reward
            
        super().memorize_transition(state, action, reward, next_state, done)

    def sample(self, batch_size):
        # proposed weights for sampling
        probs  = np.array(self.priorities)
        probs /= probs.sum()
        
        self.batch_indices = np.random.choice(len(self), batch_size, p=probs, replace=True)   # do not use replace = False, it makes O(n)
        samples = [self.buffer[idx] for idx in self.batch_indices] # seems like the fastest code for sampling!
        
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        #weights  = (probs[self.batch_indices]) ** (-self.rp_beta_by_frame(self.frames_done))
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        #weights /= (probs.min()) ** (-self.rp_beta_by_frame(self.frames_done))
       
        states, actions, rewards, next_states, dones = zip(*samples)
        #self.logger["median weight"].append(np.median(weights))
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, np.ones((batch_size))#weights

    def update_priorities(self, batch_priorities):
        """
        Updates priorities for previously sampled batch, indexes stored in self.batch_indices
        input: batch_priorities - np.array, floats, (batch_size)
        """
        diff = self.q_values.detach().cpu().numpy() - self.true_reward_to_go[self.batch_indices]
        self.priorities[self.batch_indices] = 4 / (2 + np.exp(diff) + np.exp(-diff))
        
    def see(self, state, action, reward, next_state, done):
        if len(self.buffer) > 0:        
            for g in range(state.shape[0]):
                t = (self.pos - (state.shape[0] - g)) % min(self.config.replay_buffer_capacity, len(self.buffer))
                discount = self.config.gamma
                while not self.buffer[t][4]:        # what if training was reset?
                    self.true_reward_to_go[t] += discount * reward[g]
                    discount *= self.config.gamma
                    
                    t = (t - self.env.num_envs) % min(self.config.replay_buffer_capacity, len(self.buffer))
                    
                    if (t == (self.pos - (state.shape[0] - g)) % min(self.config.replay_buffer_capacity, len(self.buffer))):
                        break
        
        super().see(state, action, reward, next_state, done)
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.priorities, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.priorities = pickle.load(mem_f)
