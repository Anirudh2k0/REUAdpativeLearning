import gymnasium
import numpy as np
from gymnasium import spaces

currTask = {"level": 1,"success":1}

class CyberDefense(gymnasium.Env):
    metadata = {'render_modes': ['human','rgb_array'],"render_fps": 4}
    def __init__(self,currTask):
        self.observation_space = spaces.Discrete(3)
        # self.observation_space = spaces.Box(low=np.array([0]),high=np.array([3]))
        self.action_space = spaces.Discrete(3)
        self.state = currTask

    def step(self,action):
        reward = 0
        done = True
        if self.state['success'] == 1:

            if action == self.state['level']+1:
                reward+=10
                observation = action
                done = True
            if self.state['level'] == action == 2:
                reward+=10
                observation = action
                done = True
            else:
                reward-=(3-action)*2
                observation = self.state['level']+1 if self.state['level']!=2 else self.state['level']

            

        else:
            if action == self.state['level']-1:
                reward+=10
                observation = action
                done = True
            if self.state['level'] == action == 0:
                reward+=10
                observation = action
                done = True
            else:
                reward-=(action+3)*2
                observation = self.state['level']-1 if self.state['level']!=0 else self.state['level']

        observation = action
        return observation,reward,done,{}
    
    def reset(self):
        observation = self.state['level']
        return observation
    
    def close(self):
        pass