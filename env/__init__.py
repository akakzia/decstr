from gym.envs.registration import register

import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])


# Robotics
# ----------------------------------------
num_blocks = 3
kwargs = {'reward_type': 'sparse'}

register(id='FetchManipulate3CloseObjects-v0',
         entry_point='env.envs:FetchManipulateCloseEnv',
         kwargs=kwargs,
         max_episode_steps=50,)

register(id='FetchManipulate3Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs=kwargs,
         max_episode_steps=100,)

register(id='FetchManipulate3AboveObjects-v0',
         entry_point='env.envs:FetchManipulateAboveEnv',
         kwargs=kwargs,
         max_episode_steps=50,)

register(id='FetchManipulate3RightCloseObjects-v0',
         entry_point='env.envs:FetchManipulateRightCloseEnv',
         kwargs=kwargs,
         max_episode_steps=50,)

register(id='FetchManipulateZoneObjects-v0',
         entry_point='env.envs:FetchManipulateZoneEnv',
         kwargs=kwargs,
         max_episode_steps=50,)


kwargs = {'reward_type': 'incremental'}

register(id='FetchManipulate3ObjectsIncremental-v0',
         entry_point='env.envs:FetchManipulateCloseEnv',
         kwargs=kwargs,
         max_episode_steps=30 * num_blocks,)
