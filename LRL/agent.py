from .utils import *
from collections import defaultdict
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

class Agent():
    """
    Basic agent interface for interacting with enviroment
    
    Args:
        env - environment
        make_env - function returning function to create a new instance of environment. Accepts seed as parameter
        threads - number of environments to create with make_envs
    """
    
    def __init__(self, config):
        self.config = config
        
        # creating environment
        if "env" in config:
            if isinstance(config["env"], VecEnv):
                self.env = config["env"]
                if self.env.num_envs > 1:
                    torch.set_num_threads(1)
            else:
                self.env = DummyVecEnv([lambda: config["env"]])
        elif "make_env" in config:
            if config.get("threads", 1) == 1:
                self.env = DummyVecEnv([config["make_env"](config["seed"])])
            else:
                self.env = SubprocVecEnv([config["make_env"](config["seed"] + i) for i in range(config["threads"])])
                torch.set_num_threads(1)
        else:
            raise Exception("Environment env or function make_env is not provided")
        
        # setting seed
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
    
    def act(self, state):
        """Responce on array of observations of enviroment"""
        return [self.env.action_space.sample() for _ in range(state.shape[0])]
    
    def see(self, state, action, reward, next_state, done):
        """Learning from new transition observed"""
        self.frames_done += 1
        
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
    
    def play(self, learn=True, render=False, record=False):
        """
        Play one game.
        input: learn, bool - whether to learn from new experience or not
        input: render, bool - whether to draw game inline
        input: record, bool - whether to store the game and show aftermath as animation
        output: cumulative reward
        """
        self.learn = learn
        
        ob = self.env.reset()
        prev_ob = ob.copy()
        R = 0        
        
        if record:
            self.record_init()
        
        for t in count():
            a = self.act(ob)
            ob, r, done, info = self.env.step(a)
            
            if learn:
                self.see(prev_ob, a, r, ob, done)
            
            R += r[0]
            prev_ob = ob.copy()
            
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
        
        if self.learn:
            self.logger["rewards"].append(R)
        
        return R
        
    def play_parallel(self, frames_limit=1000):
        """Play frames_limit frames for several games in parallel"""
        if not self.initialized:
            self.ob = self.env.reset()       
            self.prev_ob = self.ob.copy()
            self.R = np.array([0. for _ in range(self.env.num_envs)])
            self.initialized = True
        
        self.learn = True       

        for t in range(frames_limit):
            a = self.act(self.ob)
            self.ob, r, done, info = self.env.step(a)
            
            self.see(self.prev_ob, a, r, self.ob, done)
            
            self.R += r
            for res in self.R[done]:
                self.logger["rewards"].append(res)
                
            self.R[done] = 0
            self.prev_ob = self.ob.copy()
    
    def write(self, f):
        pickle.dump(self.frames_done, f)
        pickle.dump(self.logger, f)
        
    def read(self, f):
        self.frames_done = pickle.load(f)
        self.logger = pickle.load(f)
        
    def save(self, name):
        f = open(name, 'wb')
        self.write(f)
        f.close()   
        
    def load(self, name):
        f = open(name, 'rb')
        self.read(f)
        f.close()
