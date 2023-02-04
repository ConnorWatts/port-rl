from gym.envs.registration import register

from gym_port.envs.port_env import PortEnv

register(
    id=PortEnv.id,
    entry_point='gym_port.envs:PortEnv',
    max_episode_steps=1000000,
    nondeterministic=False
)