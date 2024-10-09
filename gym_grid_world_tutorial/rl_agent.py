
from typing import Optional
import numpy as np

class RlAgent:
    def __init__(self, env,rewards_f,reward_minmax,transitions_f=None,gamma=None):
        self.env = env
        self.rewards_f = rewards_f
        self.reward_minmax = reward_minmax
        self.transitions_f = transitions_f
        self.gamma = gamma
    
    def init_values_2_tabs(self):
        (nrow,ncol) = (self.env.nrow, self.env.ncol)
        #Value computations
        self.values_tabs = [ np.zeros((nrow,ncol)), np.zeros((nrow, ncol)) ]
        self.v_idx = 0
    
    def compute_next_values(self, rewards_f=None, transitions_f=None, gamma : Optional[float]=None):
        env = self.env
        if None != gamma:
            self.gamma = gamma
        gamma = self.gamma
        if None != rewards_f:
            self.rewards_f = rewards_f
        if None != transitions_f:
            self.transitions_f = transitions_f
        
        #Value computations
        prev = self.v_idx
        next = (self.v_idx + 1) % 2
        for s in range(env.state_space.n):
            row,col = env.cell_id_to_cell_coordinates(s)
            if env.is_blocked_coordinates(row,col):
                continue
            reward = self.rewards(row,col)
            s_coords = env.cell_id_to_cell_coordinates(s)
            if env.is_terminal_coordinates(row,col):
                self.values_tabs[next][row][col] = reward
            else:
                next_values = [0.0] * env.action_space.n
                for action in range(env.action_space.n):
                    for s_next in range(env.state_space.n):
                        s_next_coords = env.cell_id_to_cell_coordinates(s_next)
                        prob = self.transitions_f(s_coords,action,s_next_coords)
                        next_values[action] += prob * gamma * self.values_tabs[prev][s_next_coords[0]][s_next_coords[1]]
                self.values_tabs[next][row][col] = reward + max(next_values)
        self.v_idx = next
    
    def rewards(self,row,col):
        return self.rewards_f(self.env.desc[row,col])
    
    def values(self,row,col):
        return self.values_tabs[self.v_idx][row][col]
