from env import fetch_manipulate_env
from gym import utils

PREDICATE_THRESHOLD = 0.1

class FetchManipulateCloseEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['close'])
        utils.EzPickle.__init__(self)

class FetchManipulateEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['close', 'above'])
        utils.EzPickle.__init__(self)

class FetchManipulateAboveEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['above'])
        utils.EzPickle.__init__(self)

class FetchManipulateRightCloseEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['right_close'])
        utils.EzPickle.__init__(self)

class FetchManipulateZoneEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, 'fetch/zone_5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['inside_zone'])
        utils.EzPickle.__init__(self)
