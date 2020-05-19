import argparse
import numpy as np
from mpi4py import MPI


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='FetchManipulate3Objects-v0', help='the environment name')
    parser.add_argument('--agent', type=str, default='SAC', help='the agent name')
    parser.add_argument('--n-epochs', type=int, default=1000, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=30, help='the times to update the network')
    parser.add_argument('--biased-init', type=bool, default=True, help='use biased environment initializations')
    parser.add_argument('--automatic-buckets', type=bool, default=False, help='automatically generate buckets during training')
    parser.add_argument('--use-pairs', type=bool, default=False, help='use pairs of goals for buckets')
    parser.add_argument('--num-buckets', type=int, default=5, help='number of buckets for automatic generation')

    parser.add_argument('--evaluations', type=bool, default=True, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='the interval that save the trajectory')

    parser.add_argument('--seed', type=int, default=np.random.randint(1e6), help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='ignoramus/', help='the path to save the models')
    parser.add_argument('--folder-prefix', type=str, default='_deepsets02', help='to discriminate the model')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, help='Tune entropy')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-entropy', type=float, default=0.001, help='the learning rate of the entropy')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')

    # Curriculum learning arguments
    parser.add_argument('--curriculum-learning', type=bool, default=True, help='Use LP-based curriculum learning')
    parser.add_argument('--curriculum-eps', type=float, default=0.3, help='Prob of sampling random goal in curriculum')
    parser.add_argument('--curriculum-nu', type=float, default=0.6, help='Prob of sampling random goal in curriculum')
    parser.add_argument('--multihead-buffer', type=bool, default=True, help='use a multihead replay buffer in curriculum')
    parser.add_argument('--queue-length', type=int, default=900, help='The window size when computing competence')

    # Deep sets arguments
    parser.add_argument('--architecture', type=str, default='deepsets', help='The architecture of the networks')
    parser.add_argument('--deepsets-attention', type=bool, default=False, help='Use attention in deepsets')
    parser.add_argument('--double-critic-attention', type=bool, default=False, help='Use a different critic attention network for each critic')

    parser.add_argument('--n-test-rollouts', type=int, default=1, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')

    args = parser.parse_args()

    return args
