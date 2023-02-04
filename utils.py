from algos.ddpg import DDPG
from algos.ppo import PPO

def get_algo(config,env):

    if config["rl_algo"] == "ddpg":
        print("--Loading DDPG Agent--")
        return DDPG(config, env)

    elif config["rl_algo"] == "ppo":
        print("--Loading PPO Agent--")
        return PPO(config, env)
