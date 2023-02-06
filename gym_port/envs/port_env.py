from typing import Dict, List, Tuple, Union
import random
import gym
import numpy as np
from gym import error, spaces, utils
import pandas as pd

class PortEnv(gym.Env):

    id = 'port-v0'

    def __init__(self, stocks: list, prices: np.ndarray, returns: np.ndarray) -> None:
        super().__init__()

        self.stocks = stocks
        self.prices = prices
        self.returns = returns
        self.max_episode = len(self.prices)

        # pass this 8 in
        self.action_space = spaces.Box(0, 1, shape=(1,len(self.stocks)))
        self.observation_space = spaces.Box(0, 3, shape = self.prices[0].shape, dtype=np.float32)

        #self.step_counter = 0
        self.cum_reward = 0.0

    def step(self, action:int)-> Tuple[np.ndarray, float, bool, bool, dict]:  


        self.state = self._get_observation(self.local_step_number) 
        self.reward = self._get_reward(action,self.local_step_number)
        if self.local_step_number == self.max_episode-2:
            # log episode results
            self.done = True
        self.cum_reward += self.reward
        self.local_step_number += 1

        return self.state, self.reward, self.done, self.done,{}

        ...

    def reset(self) -> Tuple[np.ndarray]:
        self.local_step_number = 0
        self.reward = 0.0
        self.done = False
        self.state = self._get_observation(self.local_step_number)
        return self.state
        ...

    def render(self):
        ...

    def _get_reward(self, action: np.ndarray, idx: int) -> float:

        # TO DO: include transaction cost/ risk
        returns = self.returns[idx+1]
        return np.log(np.dot(action,returns)).item()



    def _get_observation(self, idx: int) -> np.ndarray:

        obs_row = self.prices.iloc[idx]
        return obs_row.to_numpy()

if __name__ == "__main__":

    # for testing
    ...