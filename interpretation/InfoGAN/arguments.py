import argparse
import numpy as np


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--input-size", type=int, default=9)
    parser.add_argument("--hidden-size", type=list, default=128)

    parser.add_argument("--k-param", type=int, default=0.6)

    parser.add_argument("--save-dir", type=str, default='data/')
    parser.add_argument("--save-model", action='store_true')

    args = parser.parse_args()

    return args
