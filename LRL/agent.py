from .logger import *
from .drawing_tools import *
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

class Agent(Logger):
    """
    Basic agent interface for interacting with enviroment
    
    Args:
        env - environment
        make_env - function returning function to create a new instance of environment. Accepts seed (int) as parameter
        threads - number of environments to create with make_envs, int
        gamma - infinite horizon protection, float, from 0 to 1
    """
    
    PARAMS = {"env", "make_env", "threads", "gamma"}
        
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
                    self.env = DummyVecEnv([config["make_env"]()])
                else:
                    self.env = SubprocVecEnv([config["make_env"]() for _ in range(config["threads"])])
            except:
                raise Exception("Error during environments creation. Try to run make_env(0) to find the bug!")
        else:
            raise Exception("Environment env or function make_env is not provided")
        
        # useful config updates
        self.config.setdefault("gamma", 0.99)
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
        self.is_recording = False       
        self.frames_done = 0
        
        self.logger_labels["rewards"] = ("episode", "reward")
        self.logger_labels["time"] = ("step", "seconds")
    
    def reset(self):
        """Called when environment is reset by force before meeting done=True"""
        pass
    
    def act(self, state):
        """
        Responce on array of observations of enviroment
        input: state - numpy array, (num_envs x observation_shape)
        output: actions - list, ints or floats, (num_envs)
        """
        return [self.env.action_space.sample() for _ in range(state.shape[0])]
    
    def see(self, state, action, reward, next_state, done):
        """
        Learning from new transition observed:
        input: state - numpy array, (num_envs x observation_shape)
        input: action - numpy array, ints or floats, (num_envs x actions_shape)
        input: reward - numpy array, (num_envs)
        input: next_state - numpy array, (num_envs x observation_shape)
        input: done - numpy array, 0 and 1, (num_envs)
        """
        self.frames_done += self.env.num_envs
        
    def record_init(self):
        """Initialize self.record for recording game"""
        self.record = defaultdict(list)
        self.record["frames"].append(self.env.render(mode = 'rgb_array'))
        
    def show_record(self):
        """
        Show animation. This function may be overloaded to run util function 
        that draws more than just a render of game
        """
        show_frames(self.record["frames"])
    
    def play(self, render=False, record=False, show_record=None):
        """
        Reset environment and play one game.
        If env is vectorized, first environment's game will be recorded.
        input: render - bool, whether to draw game inline (can be rendered in notebook)
        input: record - bool, whether to store the game and show 
        input: show_record - bool, whether to show the stored game aftermath as animation
        output: cumulative reward
        """
        if show_record is None:
            show_record = record        
        assert (not show_record) or record, "Can't show record if it is not being recorded!" 
        
        self.is_learning = False
        self.initialized = False
        self.is_recording = record
        
        ob = self.env.reset()
        R = np.zeros((self.env.num_envs), dtype=np.float32)        
        
        if record:
            self.record_init()
        
        for t in count():
            a = self.act(ob)
            ob, r, done, info = self.env.step(a)
            
            R += r
            
            if self.is_recording:                
                self.record["frames"].append(self.env.render(mode = 'rgb_array'))
                self.record["reward"].append(r)
            if render:
                clear_output(wait=True)
                plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if done[0]:
                break
                
        if self.is_recording and show_record:
            self.show_record()
        
        return R[0]
        
    def learn(self, frames=10000, plot_frequency=100):
        """
        Play frames for several games in parallel
        input: frames - int, how many observations to obtain
        input: plot_frequency - int, how often in frames to plot graphics
        """
        
        if not self.initialized:
            self.ob = self.env.reset()       
            self.prev_ob = self.ob
            self.R = np.zeros((self.env.num_envs), dtype=np.float32)
            self.reset()
            self.initialized = True
        
        self.is_learning = True
        start = time.time()
        frames_limit = (frames // self.env.num_envs) * self.env.num_envs    

        for t in range(frames_limit // self.env.num_envs):
            a = self.act(self.ob)
            
            try:
                self.ob, r, done, info = self.env.step(a)
            except:
                print("Last actions: ", a)
                raise Exception("Error during environment step. May be wrong action format?")
            
            self.see(self.prev_ob, a, r, self.ob, done)
            
            self.R += r
            for res in self.R[done]:
                self.logger["rewards"].append(res)
                self.logger["episode ends"].append(self.frames_done)
                
            self.R[done] = 0
            self.prev_ob = self.ob

            self.logger["time"].append(time.time() - start)
            start = time.time()

            if (t-1) % (plot_frequency // self.env.num_envs) == 0:
                plot_durations(self)
    
    def write(self, f):
        super().write(f)
        pickle.dump(self.frames_done, f)
        
    def read(self, f):
        super().read(f)
        self.frames_done = pickle.load(f)
