import numpy as np
import gym
import gym_object_manipulation
from utils import generate_goals
import json
import pickle as pkl


def generator(li):
    for l in li:
        yield l


def main():
    env = gym.make('FetchManipulate3Objects-v0')
    numDemos = 1
    numItr = 50
    buckets = generate_goals(nb_objects=3, sym=1, asym=1)
    demoBuffers = [{'obs': np.empty([10, 150, 55]),
                    'ag': np.empty([10, 150, 9]),
                    'g': np.empty([10, 150, 9]),
                    'actions': np.empty([10, 150, 4]),
                   } for _ in range(len(buckets.keys()))]
    idxs = [0 for _ in range(len(buckets.keys()))]
    for _ in range(numDemos):
        observations = []
        next_observations = []
        ags = []
        actions = []
        bucket = 1 #np.random.choice(np.arange(4))
        goal = buckets[bucket][np.random.choice(len(buckets[bucket]))]
        proba = 1 - 0.152 - (0.035-0.034)/0.035
        obs = env.reset_goal(np.array(goal), bucket, guide_proba=proba, train=True)
        observations.append(obs['observation'])
        ags.append(obs['achieved_goal'])
        for _ in range(10):
            env.render()
            obs, _, _, _ = env.step([0, 0, 0, -1])
        env.close()
        #itr = 1
        #itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, 0)
        """if bucket == 0:
            pair = np.random.choice(np.arange(3), size=2, replace=False)
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[0])
            itr, o, ag = pose_next_to(env, itr, observations, next_observations, ags, actions, pair[1])
            while itr < 150:
                observations.append(observations[-1])
                actions.append([0., 0., 0., 0.])
                ags.append(ags[-1])
                itr += 1
        elif bucket == 1:
            pair = np.random.choice(np.arange(3), size=3, replace=False)
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[0])
            itr, o, ag = pose_next_to(env, itr, observations, next_observations, ags, actions, pair[1])
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[2])
            itr, o, ag = pose_next_to(env, itr, observations, next_observations, ags, actions, pair[1])
            while itr < 150:
                observations.append(observations[-1])
                actions.append([0., 0., 0., 0.])
                ags.append(ags[-1])
                itr += 1
        elif bucket == 2:
            pair = np.random.choice(np.arange(3), size=2, replace=False)
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[0])
            itr, o, ag = stack_on(env, itr, observations, next_observations, ags, actions, pair[1])
            while itr < 150:
                observations.append(observations[-1])
                actions.append([0., 0., 0., 0.])
                ags.append(ags[-1])
                itr += 1
        else:
            pair = np.random.choice(np.arange(3), size=3, replace=False)
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[0])
            itr, o, ag = stack_on(env, itr, observations, next_observations, ags, actions, pair[1])
            itr, o = grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=pair[2])
            itr, o, ag = stack_on(env, itr, observations, next_observations, ags, actions, pair[1])
            while itr < 150:
                observations.append(observations[-1])
                actions.append([0., 0., 0., 0.])
                ags.append(ags[-1])
                itr += 1
        if itr == 150 and len(observations) == 150:
            actions.append([0., 0., 0., 0.])
            gs = [ags[-1] for _ in range(len(actions))]
            d = generator(buckets.items())
            key, value = next(d)
            while tuple(ag) not in value:
                key, value = next(d)
            demoBuffers[key]['obs'][idxs[key]] = observations
            demoBuffers[key]['ag'][idxs[key]] = ags
            demoBuffers[key]['g'][idxs[key]] = gs
            demoBuffers[key]['actions'][idxs[key]] = actions
            idxs[key] += 1
            print('Demonstration added to buffer number {}'.format(key))
            print('________________________________________')

    env.close()
    with open('data.json', 'w') as f:
        pkl.dump(demoBuffers, f)"""


def grasp_object(env, itr, observations, next_observations, ags, actions, target_idx=0):
    success = False
    observation = observations[-1]
    while not success and itr < 150:

        action = np.concatenate((7*(-observation[:3] + observation[10+15*target_idx:13+15*target_idx]), np.ones(1)))
        if np.linalg.norm(-observation[:3] + observation[10+15*target_idx:13+15*target_idx]) < 0.005:
            for _ in range(10):
                env.render()
                action = [0., 0., 0.4, -1]
                next_obs, r, d, info = env.step(action)
                actions.append(action)
                observation = next_obs['observation']
                itr += 1
                ag = next_obs['achieved_goal']
                next_observations.append(observation)
                observations.append(observation)
                ags.append(ag)
            success = True
        else:
            env.render()
            next_obs, r, d, info = env.step(action)
            actions.append(action)
            observation = next_obs['observation']
            ag = next_obs['achieved_goal']
            itr += 1
            next_observations.append(observation)
            observations.append(observation)
            ags.append(ag)
    return itr, observation


def stack_on(env, itr, observations, next_observations, ags, actions, target_idx=0):
    # Get above target
    success = False
    observation = observations[-1]
    ag = ags[-1]
    while not success and itr < 150:

        action = np.concatenate((7*(-observation[:2] + observation[10+15*target_idx:12+15*target_idx]),
                                 np.zeros(1), -1*np.ones(1)))
        if np.linalg.norm(-observation[:2] + observation[10+15*target_idx:12+15*target_idx]) < 0.005:
            success = True
        next_obs, r, d, info = env.step(action)
        observation = next_obs['observation']
        itr += 1
        ag = next_obs['achieved_goal']
        actions.append(action)
        next_observations.append(observation)
        observations.append(observation)
        ags.append(ag)
    # stack object
    success = False
    while not success and itr < 150:

        action = np.concatenate((np.zeros(2), [100*(-observation[2] + observation[12+15*target_idx])], -1 * np.ones(1)))
        if np.abs(-observation[2] + observation[12+15*target_idx]) < 0.07:
            for _ in range(10):

                action = [0., 0., 0., 0.3]
                next_obs, r, d, info = env.step(action)
                actions.append(action)
                observation = next_obs['observation']
                itr += 1
                ag = next_obs['achieved_goal']
                next_observations.append(observation)
                observations.append(observation)
                ags.append(ag)
            for _ in range(10):

                action = [0., 0., 0.5, 0.3]
                next_obs, r, d, info = env.step(action)
                actions.append(action)
                observation = next_obs['observation']
                itr += 1
                ag = next_obs['achieved_goal']
                next_observations.append(observation)
                observations.append(observation)
                ags.append(ag)
            success = True
        else:
            next_obs, r, d, info = env.step(action)
            observation = next_obs['observation']
            itr += 1
            ag = next_obs['achieved_goal']
            actions.append(action)
            next_observations.append(observation)
            observations.append(observation)
            ags.append(ag)
    return itr, observation, ag


def pose_next_to(env, itr, observations, next_observations, ags, actions, target_idx=0):
    # Reach
    success = False
    observation = observations[-1]
    ag = ags[-1]
    while not success and itr < 150:

        target_x_y = observation[10+15*target_idx:13+15*target_idx] + np.random.uniform(0.0, 0.1, size=3)
        action = np.concatenate((7*(-observation[:3] + target_x_y), -1*np.ones(1)))
        if np.linalg.norm(-observation[:3] + target_x_y) < 0.05:
            success = True
            for _ in range(10):

                action = [0., 0., 0.3, 0.3]
                next_obs, r, d, info = env.step(action)
                actions.append(action)
                observation = next_obs['observation']
                itr += 1
                ag = next_obs['achieved_goal']
                observations.append(observation)
                next_observations.append(observation)
                ags.append(ag)
        else:
            next_obs, r, d, info = env.step(action)
            observation = next_obs['observation']
            itr += 1
            ag = next_obs['achieved_goal']
            actions.append(action)
            observations.append(observation)
            next_observations.append(observation)
            ags.append(ag)
    return itr, observation, ag


if __name__ == "__main__":
    main()
