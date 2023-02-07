import argparse
from algos.ddpg import DDPG
#from algos.ppo import PPO
import utils

import gym
import gym_port
from gym_port import utils as gym_utils


def main(config):

    # set seeds
    
    # create environment
    env = gym.make('port-v0', **gym_utils.get_env_args(config['stocks'],config['mode'],config['input_periods']))
    
    # get algorithm - DDPG/PPO etc
    algo = get_algo(config, env)

    # run experiment
    if config['mode'] == 'Train':
        algo.train()

    elif config['mode'] == 'Test':
        algo.eval()

def get_algo(config,env):

    # TO DO: tuck this away somewhere

    if config["rl_algo"] == "ddpg":
        print("--Loading DDPG Agent--")
        return DDPG(config, env)

    elif config["rl_algo"] == "ppo":
        print("--Loading PPO Agent--")
        #return PPO(config, env)


def get_args() -> dict:

    parser = argparse.ArgumentParser(description='RL Portfolio Manager')

    parser.add_argument("--mode", type=str, help="Mode of experiment [Train,Test]", default="Train")

    # currently only supports ["III", "AAL", "ABDN", "ADM", "AHT", "ANTO", "AZN"]
    # TO DO: Dynamically download stock data for input 
    parser.add_argument("--stocks", type=list, help="List of stocks in Portfolio", default= ["III", "AAL", "ABDN", "ADM", "AHT", "ANTO", "AZN"])

    # algo parameters
    parser.add_argument("--rl_algo", type=str, help="RL Algorithm [DDPG, PPO]", default="ddpg")

    # buffer parameters
    parser.add_argument("--max_buffer_size", type=int, help="Maximum replay buffer size", default=50000)

    #training parameters 
    parser.add_argument("--total_train_steps", type=int, help="Number of training steps", default=50000)
    parser.add_argument("--warmup_steps", type=int, help="Number of steps collection experience before learning", default=10000)
    parser.add_argument("--collect_ratio", type=int, help="Number of collecting experience steps per learning steps", default=2)
    parser.add_argument("--batch_size", type=int, help="Batch size for training model", default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4,help="the learning rate of the optimizer")
    parser.add_argument("--input_periods", type=int, help="Number of input periods", default=20)

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())