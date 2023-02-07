import pandas as pd
import random
import numpy as np
import torch

from typing import Dict, List, Tuple, Union

class ReplayBuffer:
    def __init__(self,config) -> None:
        super().__init__()

        self.max_buffer_size = config['max_buffer_size']
        self.action_size = 1 + len(config['stocks'])
        self.input_periods = config['input_periods']

        # individual buffers
        self.state_buffer = np.zeros([self.max_buffer_size, self.action_size, 3 , self.input_periods])
        self.action_buffer = np.zeros([self.max_buffer_size, self.action_size])
        self.reward_buffer = np.zeros([self.max_buffer_size])
        self.done_buffer = np.zeros([self.max_buffer_size])

        self.buffer_idx = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor,\
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :

        # find the current size of the buffer as to not fetch out of bounds
        buffer_size = min(self.buffer_idx, self.max_buffer_size - 1)

        # find the set of possible samples (exclude current)
        idx_set  = set(range(buffer_size)) - set([self.buffer_idx % self.max_buffer_size])
        
        rand_idx_0 = random.sample(list(idx_set), batch_size)
        rand_idx_1 = [ (i+1) % self.max_buffer_size for i in rand_idx_0]
        rand_idx_2 = [ (i+2) % self.max_buffer_size for i in rand_idx_0]

        # get values
        states = torch.tensor(self.state_buffer[rand_idx_1]).float()
        next_states = torch.tensor(self.state_buffer[rand_idx_2]).float()
        action = torch.tensor(self.action_buffer[rand_idx_0]).float()
        next_action = torch.tensor(self.action_buffer[rand_idx_1]).float()
        reward = torch.tensor(self.reward_buffer[rand_idx_1]).float()
        done = torch.tensor(self.done_buffer[rand_idx_1]).float()

        return states, next_states, action, next_action, reward, done

    def store_transition(self, state, action, reward, done) -> None:
        store_idx_0 = self.buffer_idx % self.max_buffer_size
        store_idx_1 = (self.buffer_idx + 1) % self.max_buffer_size
        self.state_buffer[store_idx_1] = state
        self.action_buffer[store_idx_0] = action
        self.reward_buffer[store_idx_1] = reward
        self.done_buffer[store_idx_1] = done
        self.buffer_idx += 1