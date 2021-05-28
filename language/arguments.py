import argparse
import numpy as np


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--relational", default=True, help='Use a GNN based model')
    parser.add_argument("--conditional-inference", default=True, help='Perform the inference based on parsed instructions')
    # Training parameters
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=np.random.randint(1e6))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--encoder-layer-sizes", type=list, default=[128, 128])
    parser.add_argument("--decoder-layer-sizes", type=list, default=[128, 128])
    parser.add_argument("--latent-size", type=int, default=27)
    parser.add_argument("--embedding-size", type=int, default=100)

    parser.add_argument("--k-param", type=int, default=0.6)

    parser.add_argument("--save-dir", type=str, default='data/')
    parser.add_argument("--save-model", action='store_true')

    args = parser.parse_args()

    return args
