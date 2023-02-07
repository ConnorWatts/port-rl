import numpy as np
from replay_buffer import ReplayBuffer
from utils import get_actor, get_critic, get_actor_noise, get_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.optim as optim

class DDPG:
    def __init__(self, config: dict, env) -> None:
        super().__init__()

        # TO DO: pass in config[""] instead of config

        self.env = env
        self.state, self.action = self.env.reset() #X1 and w0
        self.next_action = self.action.copy()

        self.total_train_steps = config['total_train_steps']
        self.warmup_steps = config['warmup_steps']
        self.collect_ratio = config['collect_ratio']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.policy_frequency = config['policy_frequency']
        self.train_log_frequency = 100

        self.writer = SummaryWriter()

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

        self.loss_ft = get_loss(config)
        self.loss = []
        

    def train(self) -> None:

        for step in tqdm(range(self.total_train_steps)):

            self.collect_experience()

            if step > self.warmup_steps:
                self.learn_from_experience()

            self.global_step_count += 1


    def eval(self) -> None:

        # maybe in init - load_state_dict
        ...

    def collect_experience(self) -> None:

        for _ in range(self.collect_ratio):

            next_state, reward, done, _ , _= self.env.step(self.next_action) 
            self.next_action = self.sample_action(self.state, self.action) 
            self.replay_buffer.store_transition(self.state, self.action, reward, done) # store X1 w1 r2
            
            if done:
                self.state, self.action = self.env.reset()
                self.next_action = self.action.copy()
                # log episode result
            else:
                self.state, self.action = next_state.copy(), self.next_action.copy()

    def learn_from_experience(self) -> None:
        
        current_states, next_states, current_actions, next_actions, rewards, dones = \
            self.replay_buffer.sample(self.batch_size)

        self.train_critic(current_states, next_states, current_actions, next_actions, \
            rewards, dones)

        if self.global_step_count % self.policy_frequency == 0:

            self.train_actor(current_states, current_actions)

        if self.global_step_count % self.train_log_frequency == 0:
            ... # write to write

    def train_critic(self, current_states, next_states, current_actions, next_actions, \
            rewards, dones) -> None:

        with torch.no_grad():
            target_next_state_actions = self.actor_target(next_states, next_actions)
            target_next_state_q = self.critic_target(next_states, target_next_state_actions)
            next_q_value = rewards + (1 - dones) * self.gamma * target_next_state_q

        current_state_q = self.critic(current_states, current_actions)
        self.critic_loss = self.loss_ft(current_state_q, next_q_value)
        self.critic_opt.zero_grad()
        self.critic_loss.backward()
        self.critic_opt.step()


    def train_actor(self, current_states, current_actions) -> None:

        self.actor_loss = -self.critic(current_states,self.actor(current_states,current_actions))
        self.actor_opt.zero_grad()
        self.actor_loss.backward()
        self.actor_opt.step()


    def sample_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:

        # given Xt and wt-1 find wt
        # initial collection phase this is random then
        # the actor is used
        
        if self.global_step_count < self.warmup_steps:
            # random sample (be sure to norm)
            action = ...
            action = ... #normalise
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(state),torch.Tensor(action)) #.to(device))
                action += self.action_dist.sample([1])    #([self.args.noise_factor*data.shape[0]])
                action = ... #normalise - poten .cpu().numpy()
        return action