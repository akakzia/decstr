from gym.envs.registration import register

import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])


# Robotics
# ----------------------------------------
num_blocks = 3

register(id='FetchManipulate3Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs={'reward_type': 'sparse'},
         max_episode_steps=100,)

register(id='FetchManipulate3ObjectsContinuous-v0',
         entry_point='env.envs:FetchManipulateEnvContinuous',
         kwargs={'reward_type': 'incremental'},
         max_episode_steps=100,)