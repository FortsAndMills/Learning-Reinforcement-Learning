from .agent import *

class BackwardBufferAgent(Agent):
    """
    Experimental!
    
    Args:
        replay_buffer_games_capacity - size of buffer in games, int
        batch_size - size of batch on each step, int
    """
    __doc__ += Agent.__doc__
    
    def __init__(self, config):
        super().__init__(config)
        
        self.batch_size = config.get("batch_size", 32)
        self.replay_buffer_games_capacity = config.get("replay_buffer_games_capacity", 50)
        
        self.replay_buffer_nsteps = 1
        self.buffer = []
        self.pos = 0
        self.game_playing_ids = [None for i in range(self.batch_size)]
        self.sampling_index = [(0, 0) for i in range(self.batch_size)]
    
    def memorize_transition(self, state, action, reward, next_state, done, game_id):
        """Remember transition"""
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)        
        self.buffer[game_id].append((state, action, reward, next_state, done))
    
    def memorize(self, state, action, reward, next_state, done):
        """Remember batch of transitions"""
        for i, s, a, r, ns, d in zip(range(state.shape[0]), state, action, reward, next_state, done):
            game_id = self.game_playing_ids[i]
            
            if game_id is None:
                if len(self.buffer) < self.replay_buffer_games_capacity:
                    self.buffer.append([])
                else:
                    self.buffer[self.pos] = []
                    self.sampling_index = [(game_id, 0) if game_id == self.pos else (game_id, i) for game_id, i in self.sampling_index]
            
                self.game_playing_ids[i] = self.pos
                game_id = self.pos
                self.pos = (self.pos + 1) % self.replay_buffer_games_capacity
                
            self.memorize_transition(s, a, r, ns, d, game_id)
            
            if d:
                self.game_playing_ids[i] = None                
    
    def see(self, state, action, reward, next_state, done):
        self.memorize(state, action, reward, next_state, done)
        super().see(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        """
        Generate batch of given size.
        Output: state_batch - batch_size x state_dim 
        Output: action_batch - batch_size
        Output: reward_batch - batch_size
        Output: next_state_batch - batch_size x state_dim 
        Output: done_batch - batch_size
        Output: weights_batch - batch_size
        """
        assert batch_size == self.batch_size, "Backward Buffer Agent batch size must be always fixed the same"
        
        def sample_game():
            game_id = random.randint(0, len(self.buffer) - 1)
            return (game_id, len(self.buffer[game_id]) - 1)
        self.sampling_index = [(game_id, i - 1) if i > 0 else sample_game() for game_id, i in self.sampling_index]
        
        state, action, reward, next_state, done = zip(*[self.buffer[game_id][i] for game_id, i in self.sampling_index])
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
            
