from .utils import *
from collections import defaultdict
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

class Agent():
    """
    Basic agent interface for interacting with enviroment
    
    Args:
        env - environment
        make_env - function returning function to create a new instance of environment. Accepts seed (int) as parameter
        threads - number of environments to create with make_envs, int
        seed - seed for torch, numpy, torch.cuda and environments, int
    """
    
    def __init__(self, config):
        self.config = config
        
        # creating environment
        if "env" in config:
            if isinstance(config["env"], VecEnv):
                self.env = config["env"]
            else:
                self.env = DummyVecEnv([lambda: config["env"]])
        elif "make_env" in config:
            if config.get("threads", 1) == 1:
                self.env = DummyVecEnv([config["make_env"](config.get("seed", 0))])
            else:
                self.env = SubprocVecEnv([config["make_env"](config.get("seed", 0) + i) for i in range(config["threads"])])
        else:
            raise Exception("Environment env or function make_env is not provided")
            
        # ?
        if self.env.num_envs > 1:
            torch.set_num_threads(1)
        
        # setting seed. Is it really helping in reproducing experiments?!
        if "seed" in config:
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config["seed"])
        
        # useful config updates
        self.config["num_actions"] = self.env.action_space.n
        self.config["observation_shape"] = self.env.observation_space.shape
        
        # logging and initialization
        self.initialized = False          
        self.frames_done = 0
        
        self.logger = defaultdict(list)
        self.logger_labels = defaultdict(tuple)
        self.logger_labels["rewards"] = ("episode", "reward")
        self.logger_labels["fps"] = ("training epoch", "FPS")
    
    def act(self, state):
        """
        Responce on array of observations of enviroment
        input: state, numpy array of size num_envs x observation_shape
        output: actions, list of size num_envs
        """
        return [self.env.action_space.sample() for _ in range(state.shape[0])]
    
    def see(self, state, action, reward, next_state, done):
        """
        Learning from new transition observed:
        input: state, numpy array of size num_envs x observation_shape
        input: action, numpy array of ints of size num_envs
        input: reward, numpy array of size num_envs
        input: next_state, numpy array of size num_envs x observation_shape
        input: done, numpy array of size num_envs of 0 and 1
        """
        self.frames_done += self.env.num_envs
        
    def record_init(self):
        """Initialize buffer for recording game"""
        self.record = defaultdict(list)
        self.record["frames"].append(self.env.render(mode = 'rgb_array'))
    
    def record_step(self):
        """Record one step of game"""
        self.record["frames"].append(self.env.render(mode = 'rgb_array'))
        
    def show_record(self):
        """Show animation"""
        show_frames(self.record["frames"])
    
    def play(self, render=False, record=False):
        """
        Play one game.
        If env is vectorized, first environment's game will be recorded.
        input: render, bool - whether to draw game inline
        input: record, bool - whether to store the game and show aftermath as animation
        output: cumulative reward
        """
        self.is_learning = False
        
        ob = self.env.reset()
        prev_ob = ob
        R = np.zeros((self.env.num_envs), dtype=np.float32)        
        
        if record:
            self.record_init()
        
        for t in count():
            a = self.act(ob)
            ob, r, done, info = self.env.step(a)
            
            #if self.is_learning:
            #    self.see(prev_ob, a, r, ob, done)
            
            R += r
            prev_ob = ob
            
            if record:                
                self.record_step()
            if render:
                clear_output(wait=True)
                img = plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if done[0]:
                break
                
        if record:
            self.show_record()
        
        return R[0]
        
    def learn(self, frames_limit=1000):
        """
        Play frames_limit frames for several games in parallel
        input: frames_limit, how many observations to obtain
        """
        
        if not self.initialized:
            self.ob = self.env.reset()       
            self.prev_ob = self.ob
            self.R = np.zeros((self.env.num_envs), dtype=np.float32)
            self.initialized = True
        
        self.is_learning = True
        start = time.time()
        frames_limit = (frames_limit // self.env.num_envs) * self.env.num_envs    

        for t in range(frames_limit // self.env.num_envs):
            a = self.act(self.ob)
            self.ob, r, done, info = self.env.step(a)
            
            self.see(self.prev_ob, a, r, self.ob, done)
            
            self.R += r
            for res in self.R[done]:
                self.logger["rewards"].append(res)
                
            self.R[done] = 0
            self.prev_ob = self.ob
        
        self.logger["fps"].append(frames_limit / (time.time() - start))
    
    def write(self, f):
        "writing logs and data to file"
        pickle.dump(self.frames_done, f)
        pickle.dump(self.logger, f)
        
    def read(self, f):
        "reading logs and data from file"
        self.frames_done = pickle.load(f)
        self.logger = pickle.load(f)
        
    def save(self, name):
        "saving to file"
        f = open(name, 'wb')
        self.write(f)
        f.close()   
        
    def load(self, name):
        "loading from file"
        f = open(name, 'rb')
        self.read(f)
        f.close()
