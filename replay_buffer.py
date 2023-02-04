import pandas as pd
import random
import numpy as np
import torch

from typing import Dict, List, Tuple, Union

class ReplayBuffer:
    def __init__(self,config) -> None:
        super().__init__()

        self.max_buffer_size = config['max_buffer_size']
        self.state_dim = 6 #len(config['states'])

        # individual buffers
        self.state_buffer = np.zeros([self.max_buffer_size, self.state_dim])
        self.action_buffer = np.zeros([self.max_buffer_size])
        self.reward_buffer = np.zeros([self.max_buffer_size])
        self.done_buffer = np.zeros([self.max_buffer_size])

        self.buffer_idx = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :

        # find the current size of the buffer as to not fetch out of bounds
        buffer_size = min(self.buffer_idx, self.max_buffer_size - 1)

        # find the set of possible samples (exclude current)
        idx_set  = set(range(buffer_size)) - set([self.buffer_idx % self.max_buffer_size])
        rand_idx = random.sample(list(idx_set), batch_size)
        rand_idx_next = [ (i+1) % self.max_buffer_size for i in rand_idx]

        # get values
        states = torch.tensor(self.state_buffer[rand_idx]).float()
        next_states = torch.tensor(self.state_buffer[rand_idx_next]).float()
        action = torch.tensor(self.action_buffer[rand_idx]).type(torch.int64) 
        reward = torch.tensor(self.reward_buffer[rand_idx]).float()
        done = torch.tensor(self.done_buffer[rand_idx]).float()

        return states, next_states, action, reward, done

    def store_transition(self, state, next_state, action, reward, done) -> None:
        store_idx = self.buffer_idx % self.max_buffer_size
        self.state_buffer[store_idx] = state
        self.action_buffer[store_idx] = action
        self.reward_buffer[store_idx] = reward
        self.done_buffer[store_idx] = done
        self.buffer_idx += 1