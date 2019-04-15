from .replayBuffer import *

class SumTree():
    """
    Stores the priorities in sum-tree structure for effecient sampling.
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity

    def update(self, idx, p):
        """
        input: idx - id of leaf to update, int
        input: p - new priority value
        """
        assert idx < self.capacity, "SumTree overflow"
        
        idx += self.capacity - 1  # going to leaf №i
        
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:    # faster than the recursive loop
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        """
        input: v - cumulative priority of first i leafs
        output: i
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        return leaf_idx - (self.capacity - 1)
        
    def __getitem__(self, indices):
        return self.tree[indices + self.capacity - 1]

    @property
    def total_p(self):
        return self.tree[0]  # the root is sum of all priorities

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
        
        self.priorities = SumTree(self.config.replay_buffer_capacity)#np.array([])
        self.max_priority = 1.0
        self.min_priority = 1.0
                
        self.rp_beta_by_frame = lambda frame_idx: min(1.0, 
                self.config.rp_beta_start + frame_idx * (1.0 - self.config.rp_beta_start) / self.config.rp_beta_frames)
                
        self.logger_labels["median weight"] = ("training iteration", "median weight")

    def memorize_transition(self, state, action, reward, next_state, done):
        # new transition is stored with max priority
        self.priorities.update(self.pos, self.max_priority)            
        super().memorize_transition(state, action, reward, next_state, done)

    def sample(self, batch_size):
        # sample batch_size indices
        self.batch_indices = np.array([self.priorities.get_leaf(np.random.uniform(0, self.priorities.total_p)) for _ in range(batch_size)])
        
        # get transitions with these indices
        samples = [self.buffer[idx] for idx in self.batch_indices] # seems like the fastest code for sampling!
        
        # get priorities of these transitions
        batch_priorities = self.priorities[self.batch_indices] # such indexing is correct for our sumtree implementation
        
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        weights  = (batch_priorities) ** (-self.rp_beta_by_frame(self.frames_done))
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        weights /= batch_priorities.min() ** (-self.rp_beta_by_frame(self.frames_done))
       
        state, action, reward, next_state, done = zip(*samples)
        self.logger["median weight"].append(np.median(weights))
        return (Tensor(np.concatenate(state)), 
                self.ActionTensor(action), 
                Tensor(reward), 
                Tensor(np.concatenate(next_state)), 
                Tensor(done), 
                Tensor(weights))

    def update_priorities(self, batch_priorities):
        """
        Updates priorities for previously sampled batch, indexes stored in self.batch_indices
        input: batch_priorities - np.array, floats, (batch_size)
        """
        new_batch_priorities = (batch_priorities ** self.config.rp_alpha + 1e-5).clip(max=self.config.clip_priorities)
        for i, v in zip(self.batch_indices, new_batch_priorities):
            self.priorities.update(i, v) 
        self.max_priority = max(self.max_priority, new_batch_priorities.max())
        self.min_priority = min(self.min_priority, new_batch_priorities.min())
            
    def write_memory(self, mem_f):
        super().write_memory(mem_f)
        pickle.dump(self.priorities, mem_f)
        
    def read_memory(self, mem_f):
        super().read_memory(mem_f)
        self.priorities = pickle.load(mem_f)
