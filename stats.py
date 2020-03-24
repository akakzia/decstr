import numpy as np
import matplotlib.pyplot as plt


def save_plot(stats, args):
    colors = ['green', 'red', 'blue', 'indigo', 'orange']
    num_buckets = 5
    x = np.arange(0, len(stats), 1)*10
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.set_ylim(0, 1)
    for i in range(num_buckets):
        ax.plot(x, stats[:, i, 0], color=colors[i])
        ax.fill_between(x, stats[:, i, 0] - stats[:, i, 1], stats[:, i, 0] + stats[:, i, 1], color=colors[i], alpha=0.2)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.set(ylabel='success rate')
    ax.set_xlabel('Epochs')
    #plt.title('Training success rate with LP-based curriculum on 5 buckets.'
    #          ' \n predicates={close(), above()}, n_objects = 3')
    plt.grid()
    plt.legend(['Bucket {}'.format(i) for i in range(num_buckets)], fancybox=True, shadow=True, loc='lower center',
               bbox_to_anchor=(0.5, -0.25), ncol=5)
    plt.savefig('stats_{}_{}_{}.png'.format(args.architecture, args.deepsets_attention, args.double_critic_attention))


def plot_times(epochs, updates, tensorize, next_q, losses, pi, critic, worker):
    data = [epochs, updates, tensorize, next_q, losses, pi, critic]
    labels = ['epochs', 'updates', 'tensorize', 'next_q', 'losses', 'pi_step', 'critic_step']
    colors = ['royalblue', 'orangered', 'forestgreen', 'darkslategray', 'deeppink', 'goldenrod', 'peru']
    x = np.arange(0, len(epochs), 1)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    for i in range(len(data)):
        ax.plot(x, data[i], color=colors[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    plt.grid()
    plt.legend(['time_{}'.format(label) for label in labels], fancybox=True, shadow=True, loc='lower center',
               bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.savefig('timer_{}.png'.format(worker))