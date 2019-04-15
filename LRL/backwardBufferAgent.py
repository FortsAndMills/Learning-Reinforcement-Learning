from .agent import *

class BackwardBufferAgent(Agent):
    """
    Experimental! Stores transitions from different games, than samples
    game transitions in reverse order.
    
    Args:
        replay_buffer_games_capacity - size of buffer in games, int
        batch_size - size of batch on each step, int
    """
    __doc__ += Agent.__doc__
    PARAMS = Agent.PARAMS | {"replay_buffer_games_capacity", "batch_size"}    
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("batch_size", 32)
        self.config.setdefault("replay_buffer_games_capacity", 50)        
        self.config.replay_buffer_nsteps = 1
        
        self.buffer = []
        self.pos = 0
        self.game_playing_ids = [None for i in range(self.config.batch_size)]
        self.sampling_index = [(0, 0) for i in range(self.config.batch_size)]
        
    def reset(self):
        super().reset()
        self.game_playing_ids = [None for i in range(self.config.batch_size)]
    
    def memorize_transition(self, state, action, reward, next_state, done, game_id):
        """
        Remember given transition:
        input: state - numpy array, (observation_shape)
        input: action - numpy array, (action_shape)
        input: reward - float
        input: next_state - numpy array, (observation_shape)
        input: done - 0 or 1
        input: game_id - int, id of game, from which this transition was sampled
        """
        
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)        
        self.buffer[game_id].append((state, action, reward, next_state, done))
    
    def memorize(self, state, action, reward, next_state, done):
        """
        Remember batch of transitions:
        input: state - numpy array, (num_envs x observation_shape)
        input: action - numpy array of ints or floats, (num_envs x action_shape)
        input: reward - numpy array, (num_envs)
        input: next_state - numpy array, (num_envs x observation_shape)
        input: done - numpy array, 0 and 1, (num_envs)
        """
        
        for i, s, a, r, ns, d in zip(range(state.shape[0]), state, action, reward, next_state, done):
            game_id = self.game_playing_ids[i]
            
            if game_id is None:
                if len(self.buffer) < self.config.replay_buffer_games_capacity:
                    self.buffer.append([])
                else:
                    self.buffer[self.pos] = []
                    self.sampling_index = [(game_id, 0) if game_id == self.pos else (game_id, i) for game_id, i in self.sampling_index]
            
                self.game_playing_ids[i] = self.pos
                game_id = self.pos
                self.pos = (self.pos + 1) % self.config.replay_buffer_games_capacity
                
            self.memorize_transition(s, a, r, ns, d, game_id)
            
            if d:
                self.game_playing_ids[i] = None                
    
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
        assert batch_size == self.config.batch_size, "Backward Buffer Agent batch size must be always fixed the same"
        
        def sample_game():
            game_id = random.randint(0, len(self.buffer) - 1)
            return (game_id, len(self.buffer[game_id]) - 1)
        self.sampling_index = [(game_id, i - 1) if i > 0 else sample_game() for game_id, i in self.sampling_index]
        
        state, action, reward, next_state, done = zip(*[self.buffer[game_id][i] for game_id, i in self.sampling_index])
        return (Tensor(np.concatenate(state)), 
                self.ActionTensor(action), 
                Tensor(reward), 
                Tensor(np.concatenate(next_state)), 
                Tensor(done), 
                Tensor(np.ones((batch_size))))
    
    def update_priorities(self, batch_priorities):
        pass
    
    def __len__(self):
        return sum([len(game_buffer) for game_buffer in self.buffer])
    
    def read_memory(self, mem_f):
        self.buffer = pickle.load(mem_f)
        self.pos = pickle.load(mem_f)
        self.sampling_index = pickle.load(self.sampling_index)
    
    def write_memory(self, mem_f):
        pickle.dump(self.buffer, mem_f)
        pickle.dump(self.pos, mem_f)
        pickle.dump(self.sampling_index, mem_f)
    
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
