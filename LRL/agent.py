from .logger import *
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

class Agent(Logger):
    """
    Basic agent interface for interacting with enviroment
    
    Args:
        env - environment
        make_env - function returning function to create a new instance of environment. Accepts seed (int) as parameter
        threads - number of environments to create with make_envs, int
        seed - seed for torch, numpy, torch.cuda and environments, int
    """
    
    PARAMS = {"env", "make_env", "threads", "seed"}
        
    def __init__(self, config):
        super().__init__(config)
        
        # creating environment
        if "env" in config:            
            # If environment given, create DummyVecEnv shell if needed:
            if isinstance(config["env"], VecEnv):
                self.env = config["env"]
            else:
                self.env = DummyVecEnv([lambda: config["env"]])
        elif "make_env" in config:
            # Else create different environment instances.
            try:
                if config.get("threads", 1) == 1:
                    self.env = DummyVecEnv([config["make_env"](config.get("seed", 0))])
                else:
                    self.env = SubprocVecEnv([config["make_env"](config.get("seed", 0) + i) for i in range(config["threads"])])
            except:
                raise Exception("Error during environments creation. Try to run make_env(0) to find the bug!")
        else:
            raise Exception("Environment env or function make_env is not provided")
            
        # I do not understand this?..
        if self.env.num_envs > 1:
            torch.set_num_threads(1)
        
        # setting seed. Is it really helping in reproducing experiments?!
        # with cuda=async and SubprocVecEnv it can't help in principle.
        if "seed" in config:
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config["seed"])
        
        # useful config updates
        self.config["observation_shape"] = self.env.observation_space.shape
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.config["num_actions"] = self.env.action_space.n
            self.config["actions_shape"] = ()
            self.ActionTensor = LongTensor
        else:
            self.config["num_actions"] = np.array(self.env.action_space.shape).prod()
            self.config["actions_shape"] = self.env.action_space.shape
            self.ActionTensor = Tensor
        
        # logging and initialization
        self.initialized = False          
        self.frames_done = 0
        
        self.logger_labels["rewards"] = ("episode", "reward")
        self.logger_labels["fps"] = ("training epoch", "FPS")
    
    def act(self, state, record=False):
        """
        Responce on array of observations of enviroment
        input: state - numpy array, (num_envs x observation_shape)
        input: record - bool, whether to store in self.record decisions 
        output: actions - list, ints or floats, (num_envs)
        """
        return [self.env.action_space.sample() for _ in range(state.shape[0])]
    
    def see(self, state, action, reward, next_state, done):
        """
        Learning from new transition observed:
        input: state - numpy array, (num_envs x observation_shape)
        input: action - numpy array, ints or floats, (num_envs)
        input: reward - numpy array, (num_envs)
        input: next_state - numpy array, (num_envs x observation_shape)
        input: done - numpy array, 0 and 1, (num_envs)
        """
        self.frames_done += self.env.num_envs
        self.log()
        
    def record_init(self):
        """Initialize self.record for recording game"""
        self.record = defaultdict(list)
        self.record["frames"].append(self.env.render(mode = 'rgb_array'))
    
    def record_step(self):
        """Record one step of game in self.record"""
        self.record["frames"].append(self.env.render(mode = 'rgb_array'))
        
    def show_record(self):
        """
        Show animation. This function may be overloaded to run util function 
        that draws more than just a render of game
        """
        show_frames(self.record["frames"])
    
    def play(self, render=False, record=False):
        """
        Reset environment and play one game.
        If env is vectorized, first environment's game will be recorded.
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: record - bool, whether to store the game and show aftermath as animation
        output: cumulative reward
        """
        self.is_learning = False
        self.initialized = False
        
        ob = self.env.reset()
        R = np.zeros((self.env.num_envs), dtype=np.float32)        
        
        if record:
            self.record_init()
        
        for t in count():
            a = self.act(ob, record)
            ob, r, done, info = self.env.step(a)
            
            R += r
            
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
        input: frames_limit - int, how many observations to obtain
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
            
            try:
                self.ob, r, done, info = self.env.step(a)
            except:
                print(a)
                raise Exception("Broken pipe...")
            
            self.see(self.prev_ob, a, r, self.ob, done)
            
            self.R += r
            for res in self.R[done]:
                self.logger["rewards"].append(res)
                
            self.R[done] = 0
            self.prev_ob = self.ob
        
        self.logger["fps"].append(frames_limit / (time.time() - start))
    
    def write(self, f):
        super().write(f)
        pickle.dump(self.frames_done, f)
        
    def read(self, f):
        super().read(f)
        self.frames_done = pickle.load(f)
