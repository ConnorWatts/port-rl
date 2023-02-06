import numpy as np
from replay_buffer import ReplayBuffer
from utils import get_actor, get_critic, get_actor_noise
from tqdm import tqdm
import torch
import torch.optim as optim

class DDPG:
    def __init__(self, config: dict, train_env, test_env) -> None:
        super().__init__()

        # TO DO: pass in config[""] instead of config

        self.train_env = train_env
        self.test_env = test_env
        self.train_state = train_env.reset()

        self.total_train_steps = config['total_train_steps']
        self.warmup_steps = config['warmup_steps']
        self.collect_ratio = config['collect_ratio']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        # get replay buffer
        self.replay_buffer = ReplayBuffer(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

        # build actor and target 
        self.actor = get_actor(config)
        self.actor_target = get_actor(config)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # build critic and target
        self.critic = get_critic(config)
        self.critic_target = get_critic(config)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # TO DO: put in getters (for choice)
        self.critic_opt = optim.Adam(list(self.critic.parameters()), lr=self.learning_rate)
        self.actor_opt = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)

        self.action_dist = get_actor_noise(config)

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

            action = self.sample_action(self.train_state)
            next_state, reward, done, _ , _= self.train_env.step(action)
            self.replay_buffer.store_transition(self.state, next_state, action, reward, done)
            
            if done:
                self.state = self.train_env.reset()
                # log episode result
            else:
                self.state = next_state

    def learn_from_experience(self) -> None:
        
        current_states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)
        ...

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        
        if self.global_step_count == 0:
            action = ... # all money in coin 
        elif self.global_step_count < self.warmup_steps:
            # random sample (be sure to norm)
            action = ...
            action = ... #normalise
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(state)) #.to(device))
                action += self.action_dist.sample([1])    #([self.args.noise_factor*data.shape[0]])
                action = ... #normalise - poten .cpu().numpy()
        return action