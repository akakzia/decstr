import torch
from rl_modules.models import actor
from rl_modules.sac_agent import SACAgent
from arguments import get_args
import gym
import gym_object_manipulation
import numpy as np
from utils import generate_goals


# process the inputs
def normalize_goal(g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    return g_norm


def normalize(o, o_mean, o_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    return o_norm


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs


if __name__ == '__main__':
    args = get_args()
    # load the model param
    # model_path = args.save_dir + args.env_name + '_Curriculum__deepsets02' + '/model_90.pt'
    model_path = '/home/ahakakzia/model_990.pt'
    if args.architecture == 'deepsets':
        o_mean, o_std, g_mean, g_std, phi_a, phi_c, rho_a, rho_c = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        o_mean, o_std, g_mean, g_std, model, _, config = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  'max_timesteps': env._max_episode_steps,
                  }
    # create the actor network
    agent = SACAgent(args, env, env_params)
    # actor_network = actor(env_params)
    if args.architecture == 'deepsets':
        agent.model.single_phi_actor.load_state_dict(phi_a)
        agent.model.single_phi_critic.load_state_dict(phi_c)
        agent.model.rho_actor.load_state_dict(rho_a)
        agent.model.rho_critic.load_state_dict(rho_c)
        #agent.model.attention.load_state_dict(att)
    else:
        agent.actor_network.load_state_dict(model)
        agent.actor_network.eval()
        agent.configuration_network.load_state_dict(config)

    s = 0
    buckets = generate_goals(nb_objects=3, sym=1, asym=1)
    for i in range(args.demo_length):
        goal = buckets[6][np.random.choice(len(buckets[6]))]
        observation = env.reset_goal(np.array(goal), eval=False)
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        ag = observation['achieved_goal']
        for t in range(30):
            env.render()
            with torch.no_grad():
                g_norm = torch.tensor(normalize_goal(g, g_mean, g_std, args), dtype=torch.float32).unsqueeze(0)
                ag_norm = torch.tensor(normalize_goal(ag, g_mean, g_std, args), dtype=torch.float32).unsqueeze(0)
                if agent.architecture == 'deepsets':
                    obs_tensor = torch.tensor(normalize(obs, o_mean, o_std, args), dtype=torch.float32).unsqueeze(0)
                    agent.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, eval=True)
                    action = agent.model.pi_tensor.numpy()[0]
                elif agent.architecture == 'disentangled':
                    z_ag = agent.configuration_network(ag_norm)[0]
                    z_g = agent.configuration_network(g_norm)[0]
                    input_tensor = torch.tensor(np.concatenate([normalize(obs, o_mean, o_std, args), z_ag, z_g]),
                                                dtype=torch.float32).unsqueeze(0)
                    action = agent._select_actions(input_tensor, eval=True)
                else:
                    input_tensor = agent._preproc_inputs(obs, ag, g)  # PROCESSING TO CHECK
                    action = agent._select_actions(input_tensor, eval=True)
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            if info['is_success']:
                s += 1
                break
            obs = observation_new['observation']
            ag = observation_new['achieved_goal']
        print('Goal: {} | the episode is: {}, is success: {}'.format(g, i, info['is_success']))
        print('Achieved goal was: {}'.format(observation_new['achieved_goal']))
    print('Success rate = {}'.format(s/args.demo_length))
