from typing import Dict, List, Tuple, Union
import random
import gym
import numpy as np
from gym import error, spaces, utils
import pandas as pd

class PortEnv(gym.Env):

    id = 'port-v0'

    def __init__(self, check: int) -> None:
        super().__init__()

        # pass this 8 in
        self.action_space = spaces.Box(0, 1, shape=(1,8))
        self.observation_space = spaces.Box(0, 1, dtype=np.float32)

    def step(self, action:int)-> Tuple[np.ndarray, float, bool, bool, dict]:   
        ...

    def reset(self) -> Tuple[np.ndarray]:
        ...

    def render(self):
        ...

if __name__ == "__main__":

    # for testing
    ...