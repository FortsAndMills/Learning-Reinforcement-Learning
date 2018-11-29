from .agent import *

class ReplayBufferAgent(Agent):
    """
    Replay Memory storing all transitions seen with basic uniform batch sampling.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        replay_buffer_capacity - size of buffer, int
    """
    __doc__ += Agent.__doc__
    PARAMS = Agent.PARAMS | {"replay_buffer_capacity"}
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("replay_buffer_capacity", 100000)
        
        self.config.replay_buffer_nsteps = 1
        self.buffer = []
        self.pos = 0
    
    def memorize_transition(self, state, action, reward, next_state, done):
        """
        Remember given transition:
        input: state - numpy array, (observation_shape)
        input: action - float or int
        input: reward - float
        input: next_state - numpy array, (observation_shape)
        input: done - 0 or 1
        """
        
        # preparing for concatenation into batch in future
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        # this turned out to be the quickest way of working with experience memory
        if len(self) < self.config.replay_buffer_capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.pos = (self.pos + 1) % self.config.replay_buffer_capacity
    
    def memorize(self, state, action, reward, next_state, done):
        """
        Remember batch of transitions:
        input: state - numpy array, (num_envs x observation_shape)
        input: action - numpy array of ints or floats, (num_envs)
        input: reward - numpy array, (num_envs)
        input: next_state - numpy array, (num_envs x observation_shape)
        input: done - numpy array, 0 and 1, (num_envs)
        """
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.memorize_transition(s, a, r, ns, d)
    
    def see(self, state, action, reward, next_state, done):
        self.memorize(state, action, reward, next_state, done)
        super().see(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        """
        Generate batch of given size.
        input: batch_size - int
        output: state_batch - numpy array, (batch_size x state_dim)
        output: action_batch - numpy array, (batch_size)
        output: reward_batch - numpy array, (batch_size)
        output: next_state_batch - numpy array, (batch_size x state_dim)
        output: done_batch - numpy array, (batch_size)
        output: weights_batch - numpy array, (batch_size)
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, np.ones((batch_size))
    
    def update_priorities(self, batch_priorities):
        pass
    
    def __len__(self):
        return len(self.buffer)
    
    def read_memory(self, mem_f):
        self.buffer = pickle.load(mem_f)
        self.pos = pickle.load(mem_f)
    
    def write_memory(self, mem_f):
        pickle.dump(self.buffer, mem_f)
        pickle.dump(self.pos, mem_f)
    
    def save(self, name, save_replay_memory=False):
        super().save(name)
        
        if save_replay_memory:
            mem_f = open(name + "-memory", 'wb')
            self.write_memory(mem_f)
            mem_f.close() 
        
    def load(self, name, load_replay_memory=False):
        super().load(name)
        
        if load_replay_memory:
            mem_f = open(name + "-memory", 'rb')
            self.read_memory(mem_f)
            mem_f.close()
            
