import numpy as np
from replay_buffer import ReplayBuffer
from tqdm import tqdm
import torch

class DDPG:
    def __init__(self, config: dict, env) -> None:
        super().__init__()

        self.env = env
        self.state = env.reset()

        self.total_train_steps = config['total_train_steps']
        self.warmup_steps = config['warmup_steps']
        self.collect_ratio = config['collect_ratio']
        self.batch_size = config['batch_size']

        # pass in config[""] instead of config
        self.replay_buffer = ReplayBuffer(config)

        # TO DO: 
        self.actor = ...
        self.actor_target = ...

        self.critic = ...
        self.critic_target = ...

        self.critic_opt = ...
        self.actor_opt = ...

        self.action_dist = ... # get function

        self.global_step_count = 0
        self.learn_step_count = 0
        self.loss = []
        

    def train(self) -> None:

        for step in tqdm(range(self.total_train_steps)):

            self.collect_experience()

            if step > self.warmup_steps:
                self.learn_from_experience()

            self.global_step_count += 1


    def eval(self) -> None:
        ...

    def collect_experience(self) -> None:

        for _ in range(self.collect_ratio):

            action = self.sample_action(self.state)
            next_state, reward, done, _ , _= self.env.step(action)
            self.replay_buffer.store_transition(self.state, next_state, action, reward, done)
            
            if done:
                self.state = self.env.reset()
                # log episode result
            else:
                self.state = next_state

    def learn_from_experience(self) -> None:
        
        current_states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)
        ...

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        
        if self.global_step_count < self.warmup_steps:
            # random sample (be sure to norm)
            action = ...
            action = ... #normalise
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(state)) #.to(device))
                action += ... # plus self.action_dist
                action = ... #normalise - poten .cpu().numpy()
        return action