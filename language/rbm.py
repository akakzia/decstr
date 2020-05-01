# from https://github.com/FengZiYjun/Restricted-Boltzmann-Machine
# python: 3.6
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing.dummy import Pool as ThreadPool


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, m_observe=784):
        """Initialize model.
        Args:
            n_hidden: int, the number of hidden units
            m_observe: int, the number of visible units
        """
        self.n_hidden = n_hidden
        self.m_visible = m_observe

        self.visible = None
        self.weight = np.random.rand(self.m_visible, self.n_hidden)  # [m, n]
        self.a = np.random.rand(self.m_visible, 1)  # [m, 1]
        self.b = np.random.rand(self.n_hidden, 1)  # [n, 1]

        self.alpha = 0.01
        self.avg_energy_record = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, data, epochs=2):
        """train the RBM
        Args:
            data: numpy ndarray of shape [N, m], representing N sample with m visible units
            epochs: int, the total number of epochs in training
        """

        self.avg_energy_record.clear()
        self.visible = data
        self.__contrastive_divergence(self.visible, epochs)

        print("training finished")

    def __forward(self, v):
        h_dist = self.sigmoid(
            np.matmul(np.transpose(self.weight), v) + self.b)  # [n, 1]
        return self.__sampling(h_dist)  # [n, 1]

    def __backward(self, h):
        v_dist = self.sigmoid(np.matmul(self.weight, h) + self.a)  # [m, 1]
        return self.__sampling(v_dist)  # [m, 1]

    def __sampling(self, distribution):
        dim = distribution.shape[0]
        true_idx = np.random.uniform(0, 1, dim).reshape(dim, 1) <= distribution
        sampled = np.zeros((dim, 1))
        sampled[true_idx] = 1  # [n, 1]
        return sampled

    def __CD_1(self, v_n):
        v_n = v_n.reshape(-1, 1)
        h_sampled = self.__forward(v_n)
        v_sampled = self.__backward(h_sampled)
        h_recon = self.__forward(v_sampled)

        self.weight += self.alpha * \
            (np.matmul(v_n, np.transpose(h_sampled)) -
             np.matmul(v_sampled, np.transpose(h_recon)))
        self.a += self.alpha * (v_n - v_sampled)
        self.b += self.alpha * (h_sampled - h_recon)

        self.energy_list.append(self._energy(v_n, h_recon))

    def __contrastive_divergence(self, data, max_epoch):
        train_time = []
        for t in range(max_epoch):
            np.random.shuffle(data)
            self.energy_list = []

            start = time.time()
            pool = ThreadPool(5)
            pool.map(self.__CD_1, data)
            end = time.time()

            avg_energy = np.mean(self.energy_list)
            print("[epoch {}] takes {:.2f}s, average energy={}".format(
                t, end - start, avg_energy))
            self.avg_energy_record.append(avg_energy)
            train_time.append(end - start)
        print("Average Training Time: {:.2f}".format(np.mean(train_time)))

    def _energy(self, visible, hidden):
        return - np.inner(self.a.flatten(), visible.flatten()) - np.inner(self.b.flatten(), hidden.flatten()) \
            - np.matmul(np.matmul(visible.transpose(), self.weight), hidden)

    def energy(self, v):
        hidden = self.__forward(v)
        return self._energy(v, hidden)

    def __Gibbs_sampling(self, v_init, num_iter=10):
        v_t = v_init.reshape(-1, 1)
        for t in range(num_iter):
            h_dist = self.sigmoid(
                np.matmul(np.transpose(self.weight), v_t) + self.b)  # [n, 1]
            h_t = self.__sampling(h_dist)  # [n, 1]

            v_dist = self.sigmoid(
                np.matmul(self.weight, h_t) + self.a)  # [m, 1]
            v_t = self.__sampling(v_dist)  # [m, 1]

        return v_t, h_t

    def sample(self, num_iter=10, v_init=None):
        """Sample from trained model.
        Args:
            num_iter: int, the number of iterations used in Gibbs sampling
            v_init: numpy ndarray of shape [m, 1], the initial visible units (default: None)
        Return:
            v: numpy ndarray of shape [m, 1], the visible units reconstructed from RBM.
        """
        if v_init is None:
            v_init = np.random.rand(self.m_visible, 1)
        v, h = self.__Gibbs_sampling(v_init, num_iter)
        return v


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(mnist.shape)

    # construct rbm model
    rbm = RBM(100, 28 * 28)

    print("Start RBM training.")
    # train rbm model using mnist
    rbm.train(mnist[:200], epochs=10)
    print("Finish RBM training.")

    # sample from rbm model
    v = rbm.sample(num_iter=200, v_init=mnist[0])
    plt.imshow(v.reshape((28, 28)), cmap="gray")
    plt.show()