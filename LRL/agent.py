from .utils import *

class Agent():
    """
    Basic agent interface for interacting with enviroment
    
    Args:
        env - environment
    """
    
    def __init__(self, env):
        self.env = env
        self.frames_done = 0
        self.rewards_log = []
    
    def random_act(self):
        return self.env.action_space.sample()
    
    def act(self, state):
        """Responce on observation of enviroment"""
        return self.random_act()
    
    def see(self, state, action, reward, next_state, done):
        """Learning from new transition observed"""
        self.frames_done += 1
        
    def record_init(self):
        """Initialize buffer for recording game"""
        self.frames = []
        self.frames.append(self.env.render(mode = 'rgb_array'))
    
    def record_step(self):
        """Record one step of game"""
        self.frames.append(self.env.render(mode = 'rgb_array'))
        
    def show_record(self):
        """Show animation"""
        show_frames(self.frames)
    
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
        prev_ob = ob
        R, r = 0, 0        
        
        if record:
            self.record_init()
        
        for t in count():
            a = self.act(ob)
            ob, r, done, info = self.env.step(a)
            
            if learn:
                self.see(prev_ob, a, r, ob, done)
            
            R += r
            prev_ob = ob
            
            if record:                
                self.record_step()
            if render:
                clear_output(wait=True)
                img = plt.imshow(self.env.render(mode='rgb_array'))
                plt.show()
            
            if done or ("died" in info and info["died"]):
                break
                
        if record:
            self.show_record()
        
        if self.learn:
            self.rewards_log.append(R)
        
        return R
    
    def write(self, f):
        pickle.dump(self.frames_done, f)
        pickle.dump(self.rewards_log, f)
        
    def read(self, f):
        self.frames_done = pickle.load(f)
        self.rewards_log = pickle.load(f)
        
    def save(self, name):
        f = open(name, 'wb')
        self.write(f)
        f.close()   
        
    def load(self, name):
        f = open(name, 'rb')
        self.read(f)
        f.close()
