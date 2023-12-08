import gym
from .gridworld import TwoDGridWorld

def get_env(env_type, *kwargs):
    if env_type == 'gridworld':
        return TwoDGridWorld()
    else:
        return gym.make(env_type)