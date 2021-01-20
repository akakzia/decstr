import numpy as np
import os
import itertools

from env import rotations, robot_env, utils


def objects_distance(x, y):
    """
    A function that returns the euclidean distance between two objects x and y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x - y)


def above(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x[:2] - y[:2]) < 0.05 and 0.06 > x[2] - y[2] > 0.03


class FetchManipulateEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, num_blocks, n_substeps, gripper_extra_height, block_gripper,
        target_in_the_air, target_offset, obj_range, target_range, predicate_threshold,
            initial_qpos, reward_type, predicates, p_coplanar, p_stack_two, p_grasp,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            num_blocks: number of block objects in environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type: Only 'sparse' rewards are implemented
            predicates: 'above' and 'close'
            p_coplanar: The probability of initializing all blocks without stacks
            p_stack_two: The probability of having exactly one stack of two given that there is at least one stack
            p_grasp: The probability of having a block grasped at initialization
        """
        
        self.target_goal = None
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.predicate_threshold = predicate_threshold
        self.predicates = predicates
        self.num_predicates = len(self.predicates)
        self.reward_type = reward_type
        self.p_coplanar = p_coplanar
        self.p_stack_two = p_stack_two
        self.p_grasp = p_grasp

        self.goal_size = 0

        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        self.object_inds = [list(range(10 + i * 15, 10 + (i + 1) * 15)) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000

        super(FetchManipulateEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # Heatmap Generation
    # ----------------------------

    def set_location_record_name(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_location_record()
        self.location_record_write_dir = write_dir
        self.location_record_prefix = prefix
        self.location_record_file_number = 0

        return True

    def flush_location_record(self, create_new_empty_record=True):
        if self.location_record is not None and self.location_record_steps_recorded > 0:
            write_file = os.path.join(self.location_record_write_dir,"{}_{}".format(self.location_record_prefix,
                                                                           self.location_record_file_number))
            np.save(write_file, self.location_record[:self.location_record_steps_recorded])
            self.location_record_file_number += 1
            self.location_record_steps_recorded = 0

        if create_new_empty_record:
            self.location_record = np.empty(shape=(self.location_record_max_steps, 3), dtype=np.float32)

    def log_location(self, location):
        if self.location_record is not None:
            self.location_record[self.location_record_steps_recorded] = location
            self.location_record_steps_recorded += 1

            if self.location_record_steps_recorded >= self.location_record_max_steps:
                self.flush_location_record()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        assert self.reward_type == 'sparse', "only sparse reward type is implemented."
        reward = (achieved_goal == goal).all().astype(np.float32)
        return reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.log_location(location=self.sim.data.get_site_xpos('robot0:grip'))
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
    
    def _get_configuration(self, positions):
        """
        This functions takes as input the positions of the objects in the scene and outputs the corresponding semantic configuration
        based on the environment predicates
        """
        close_config = np.array([])
        above_config = np.array([])
        if "close" in self.predicates:
            object_combinations = itertools.combinations(positions, 2)
            object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

            close_config = np.array([(distance <= self.predicate_threshold).astype(np.float32)
                                     for distance in object_rel_distances])
        if "above" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]), (positions[0], positions[2]),
                                       (positions[2], positions[0]), (positions[1], positions[2]), (positions[2], positions[1])]
            else:
                raise NotImplementedError

            above_config = np.array([int(above(obj[0], obj[1])) for obj in object_permutations]).astype(np.float32)
        
        res = np.concatenate([close_config, above_config])
        return res

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        objects_positions = []

        for i in range(self.num_blocks):

            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            objects_positions = np.concatenate([
                objects_positions, object_i_pos.ravel()
            ])

        objects_positions = objects_positions.reshape(self.num_blocks, 3)
        object_combinations = itertools.combinations(objects_positions, 2)
        object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

        self.goal_size = len(object_rel_distances)
        
        achieved_goal = self._get_configuration(objects_positions)

        achieved_goal = np.squeeze(achieved_goal)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.target_goal.copy(),
            'achieved_goal_binary': achieved_goal.copy(),
            'desired_goal_binary': self.target_goal.copy()}

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        self.sim.forward()

    def reset(self):
        # Usual reset overriden by reset_goal, that specifies a goal
        return self.reset_goal(np.zeros([9]), False)

    def _generate_valid_goal(self):
        raise NotImplementedError

    def _sample_goal(self):
        self.target_goal = np.random.randint(2, size=9).astype(np.float32)
        
        return self.target_goal

    def _is_success(self, achieved_goal, desired_goal):
        return (achieved_goal == desired_goal).all()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

            # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def reset_goal(self, goal, biased_init=False):
        """
        This function resets the environment and target the goal given as input
        Args:
            goal: The semantic configuration to target
            biased_init: Whether or not to initialize the blocks in non-trivial configuration
        """

        self.target_goal = goal

        self.sim.set_state(self.initial_state)

        if biased_init and np.random.uniform() > self.p_coplanar:
            if np.random.uniform() > self.p_stack_two:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
                z_stack = [0.525, 0.475, 0.425]
            else:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
                z_stack = [0.475, 0.425]

            pos_stack = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                if i in stack:
                    object_qpos[2] = z_stack[stack.index(i)]
                    object_qpos[:2] = pos_stack.copy()

                else:
                    # place third object at least 0.05 away from other cubes
                    object_qpos[2] = 0.425
                    counter = 0
                    while counter < 100:
                        counter += 1
                        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                             self.obj_range,
                                                                                             size=2)
                        if np.linalg.norm(object_xpos - pos_stack) > (np.sqrt(2) * 0.05):
                            break
                    object_qpos[:2] = object_xpos.copy()
                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

        else:
            # Reset with all coplanar blocks and make sure to avoid unstable cases (blocks initialized at the same spot)
            stack = None
            # place cubes away from each other
            obj_placed = 0
            positions = []
            over = False
            while not over:
                over = True
                counter = 0
                while obj_placed < len(self.object_names):
                    counter += 1
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[obj_placed]))
                    assert object_qpos.shape == (7,)
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos
                    to_place = True
                    for p in positions:
                        if np.linalg.norm(object_xpos - p) < (np.sqrt(2) * 0.05):
                            to_place = False
                            break
                    if to_place:
                        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[obj_placed]), object_qpos)
                        positions.append(object_xpos.copy())
                        obj_placed += 1
                    if counter > 100:
                        # safety net to be sure we find positions
                        over = False
                        break
        if biased_init and np.random.rand() < self.p_grasp:
            ids = list(range(self.num_blocks))
            # do not grasp base of stack
            if stack:
                for s in stack[1:]:
                    ids.remove(s)
            idx_grasp = np.random.choice(ids)
            self._grasp(idx_grasp)

        self.sim.forward()
        obs = self._get_obs()
        return obs

    def _grasp(self, i):
        obj = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
        obj[:3] = self.sim.data.get_site_xpos('robot0:grip')
        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), obj.copy())
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.0240)
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.0240)
