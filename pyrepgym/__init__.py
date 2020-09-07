from gym.envs.registration import register

from pyrepgym.envs import PyRepEnv
register(id='PyRepEnv-v0', entry_point='pyrepgym.envs.PyRepEnv:PyRepEnv')
