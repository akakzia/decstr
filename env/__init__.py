from gym.envs.registration import register

import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])


# Robotics
# ----------------------------------------
num_blocks = 3
kwargs = {'reward_type': 'sparse'}

register(id='FetchManipulate3Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs=kwargs,
         max_episode_steps=100,)

register(id='FetchManipulate3ObjectsContinuous-v0',
         entry_point='env.envs:FetchManipulateEnvContinuous',
         kwargs=kwargs,
         max_episode_steps=100,)
