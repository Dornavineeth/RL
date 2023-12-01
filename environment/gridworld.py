# import necessary libraries
import gym
import numpy as np
from gym import spaces
from copy import deepcopy
# custom 2d grid world enviroment which extends gym.Env
class TwoDGridWorld(gym.Env):
    """
        - a size x size grid world which agent can ba at any cell other than terminal cell
        - terminal cell is set to be the last cell or bottom right cell in the grid world
        - 5x5 grid world example where X is the agent location and O is the tremial cell
          .....
          .....
          ..X..
          .....
          ....O -> this is the terminal cell where this is agent headed to  
        - Reference : https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['console']}
    
    # actions available 
    UP   = 0
    LEFT = 1
    DOWN = 2
    RIGHT= 3
    BREAK = 4
    
    def __init__(self, size, terminal_reward=10, water_reward= -10):
        super(TwoDGridWorld, self).__init__()
        
        self.size      = size # size of the grid world
        self.end_state = size*size - 1 # bottom right or last cell
        
        # randomly assign the inital location of agent
        self.agent_position = 0 #np.random.randint( (self.size*self.size) - 1 )
        
        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)
        
        # set the observation space to (1,) to represent agent position in the grid world 
        # staring from [0,size*size)
        self.observation_space = spaces.Box(low=0, high=size*size, shape=(1,), dtype=np.uint8)

        self.terminal_reward = terminal_reward
        self.water_reward = water_reward

        self.water = [22]
        self.obstacle = [12, 17]

        self.grid = []
        for x in range(self.size):
            row = []
            for y in range(self.size):
                row.append(".")
            self.grid.append(row)

        for w in self.water:
            row = w//self.size
            col = w%self.size
            self.grid[row][col] = 'W'
        
        for ob in self.obstacle:
            row = ob//self.size
            col = ob%self.size
            self.grid[row][col] = 'O'
        
        self.trans = {
            "forward": 0.8,
            "right": 0.05,
            "left": 0.05,
            "break": 0.1
        }

    def sample_action(self, action, sample=False):
        probs = [0,0,0,0,0]
        actions = [self.DOWN, self.RIGHT, self.LEFT, self.UP, self.BREAK]
        if action == self.RIGHT:
            probs = [ self.trans['right'], self.trans['forward'], 0.0, self.trans['left'], self.trans['break']]
        elif action == self.LEFT:
            probs = [self.trans['left'], 0.0, self.trans['forward'], self.trans['right'], self.trans['break']]
        elif action == self.DOWN:
            probs = [self.trans['forward'], self.trans['left'], self.trans['right'], 0.0, self.trans['break']]
        elif action == self.UP:
            probs = [0.0, self.trans['right'], self.trans['left'], self.trans['forward'], self.trans['break']]
        if sample:
            action = np.random.choice(actions,p=probs)
            return {k:v for k,v in zip(actions, probs)}, action
        return  {k:v for k,v in zip(actions, probs)}
        

    def step(self, action):
        print(action)
        info = {} # additional information
        _, action = self.sample_action(action, sample=True)
        
        reward = 0
        done = False
        
        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        if action == self.UP:
            if row > 0:
                row -= 1
        elif action == self.DOWN:
            if row < self.size-1:
                row += 1
        elif action == self.LEFT:
            if col > 0:
                col -= 1
        elif action == self.RIGHT:
            if col < self.size-1:
               col += 1
        agent_position = row*self.size + col 
        if agent_position in self.water:
            reward = self.water_reward
            self.agent_position = agent_position
        elif agent_position in self.obstacle:
            reward = 0
        elif agent_position == self.end_state:
            reward = self.terminal_reward
            done = True
            self.agent_position = agent_position
        else:
            reward = 0
            self.agent_position = agent_position

        # if action == self.UP:
        #     if row != 0:
        #         self.agent_position -= self.size
        #     else:
        #         reward = 0
        # elif action == self.LEFT:
        #     if col != 0:
        #         self.agent_position -= 1
        #     else:
        #         reward = 0
        # elif action == self.DOWN:
        #     if row != self.size - 1:
        #         self.agent_position += self.size
        #     else:
        #         reward = 0
        # elif action == self.RIGHT:
        #     if col != self.size - 1:
        #         self.agent_position += 1
        #     else:
        #         reward = 0
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        
        return np.array([self.agent_position]).astype(np.uint8), reward, done, info
    
    def print_grid(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                print(grid[i][j], end='\t')
            print()

    def render(self, mode='console'):
        '''
            render the state
        '''
        if mode != 'console':
          raise NotImplementedError()
        
        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        grid = deepcopy(self.grid)
        grid[row][col] = 'X'
        self.print_grid(grid)

    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        self.agent_position = 0 # np.random.randint( (self.size*self.size) - 1 )
        
        return np.array([self.agent_position]).astype(np.uint8)
    
    def close(self):
        pass

if __name__ == "__main__":
    env = TwoDGridWorld(5)
    env.render()