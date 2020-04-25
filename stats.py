import numpy as np
import matplotlib.pyplot as plt


def save_plot(stats, args):
    colors = ['forestgreen', 'tomato', 'darkcyan', 'mediumpurple', 'darkorange', 'dimgray', 'gold']
    num_buckets = stats.shape[1]
    x = np.arange(0, len(stats), 1)*args.save_freq*args.n_cycles*args.num_rollouts_per_mpi*50*24
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.set_ylim(0, 1)
    for i in range(num_buckets):
        ax.plot(x, stats[:, i, 0], color=colors[i])
        ax.fill_between(x, stats[:, i, 0] - stats[:, i, 1], stats[:, i, 0] + stats[:, i, 1], color=colors[i], alpha=0.2)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.set(ylabel='success rate')
    ax.set_xlim(0, args.n_epochs*args.n_cycles*args.num_rollouts_per_mpi*50*24)
    ax.set_xlabel('Time steps')
    plt.grid()
    plt.legend(['Bucket {}'.format(i+1) for i in range(num_buckets)], fancybox=True, shadow=True, loc='lower center',
               bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.savefig('stats_{}_Attention_{}_Curriculum_{}.png'.format(args.architecture, args.deepsets_attention, args.curriculum_learning))
    plt.cla()
    plt.close(fig)


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