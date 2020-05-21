import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
from utils import get_stat_func, generate_goals, generate_all_goals_in_goal_space

font = {'size': 50}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]


RESULTS_PATH = '/home/flowers/Desktop/Scratch/sac_curriculum/results/'
SAVE_PATH = '/home/flowers/Desktop/Scratch/sac_curriculum/results/plots/'
TO_PLOT = ['init_study']

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 7
MARKERSIZE = 25
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 's', 'v', 'X', 'D', 'P', "*"]
FREQ = 20
NB_BUCKETS = 5
NB_EPS_PER_EPOCH = 2400
NB_VALID_GOALS = 35
# EPISODES = np.arange(0, N_EPOCHS * 2400, 2400)
# X_SIZE = len(EPISODES)
# FREQ = 10
# SUB_EPISODES = np.arange(0, N_EPOCHS * 2400, 2400 * FREQ)
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None, factor=1):
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_four_figs(xlabels=None, ylabels=None, xlims=None, ylims=None):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), frameon=False)
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=4, direction='in', length=10, labelsize='15', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax], fontsize='15')
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax], fontsize='15')
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')

def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        for run in list_runs:
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            nb_epochs = len(data_run)
            if nb_epochs > max_len:
                max_len = nb_epochs
    return max_len, max_seeds


def plot_c_lp_p_sr(experiment_path, true_buckets=True):
    conditions = os.listdir(experiment_path)
    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for run in list_runs:
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            x_eps = np.arange(0, len(data_run) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
            x_eps = np.arange(0, len(data_run), FREQ)
            x = np.arange(0, len(data_run), FREQ)
            artists, axs = setup_four_figs(xlabels=['Epochs'] * 4,
                                           # xlabels=['Episodes (x$10^3$)'] * 4,
                                           ylabels=['SR', 'C', 'LP', 'P'],
                                           xlims=[(0, x_eps[-1])] * 4,
                                           ylims=[(0, 1), (0, 1), None, (0, 1)])
            if true_buckets:
                buckets = generate_goals(nb_objects=3, sym=1, asym=1)
                all_goals = generate_all_goals_in_goal_space().astype(np.float32)
                valid_goals = []
                for k in buckets.keys():
                    valid_goals += buckets[k]
                valid_goals = np.array(valid_goals)
                all_goals = np.array(all_goals)
                num_goals = all_goals.shape[0]
                all_goals_str = [str(g) for g in all_goals]
                # initialize dict to convert from the oracle id to goals and vice versa.
                # oracle id is position in the all_goal array
                g_str_to_oracle_id = dict(zip(all_goals_str, range(num_goals)))
                valid_goals_oracle_ids = np.array([g_str_to_oracle_id[str(vg)] for vg in valid_goals])

                bucket_ids = dict()
                for k in buckets.keys():
                    bucket_ids[k] = np.array([g_str_to_oracle_id[str(np.array(g))] for g in buckets[k]])
                    id_in_valid = [int(np.argwhere(valid_goals_oracle_ids == i).flatten()) for i in bucket_ids[k]]
                    sr = np.mean([data_run['Eval_SR_{}'.format(i)] for i in id_in_valid], axis=0)
                    axs[0].plot(x_eps, sr[x], color=colors[k], marker=MARKERS[k], markersize=MARKERSIZE//3, linewidth=LINEWIDTH//2)
            else:
                T = len(data_run['Eval_SR_1'])
                SR = np.zeros([NB_BUCKETS, T])
                for t in range(T):
                    for i in range(NB_BUCKETS):
                        ids = []
                        for g_id in range(35):
                            if data_run['{}_in_bucket'.format(g_id)][t] == i:
                                ids.append(g_id)
                        values = [data_run['Eval_SR_{}'.format(g_id)][t] for g_id in ids]
                        SR[i, t] = np.mean(values)

                for i in range(SR.shape[0]):
                    axs[0].plot(x_eps, SR[i][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE//3, linewidth=LINEWIDTH//2)


            for i in range(NB_BUCKETS):
                axs[1].plot(x_eps, data_run['B_{}_C'.format(i)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE//3, linewidth=LINEWIDTH//2)
                axs[2].plot(x_eps, data_run['B_{}_LP'.format(i)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE//3, linewidth=LINEWIDTH//2)
                axs[3].plot(x_eps, data_run['B_{}_p'.format(i)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE//3, linewidth=LINEWIDTH//2)
            leg = axs[0].legend(['B{}'.format(i) for i in range(1, 6)],
                             loc='upper center',
                             bbox_to_anchor=(0.5, 1.35),
                             ncol=5,
                             fancybox=True,
                             shadow=True,
                             prop={'size': 15, 'weight': 'bold'},
                             markerscale=1)
            artists += (leg,)
            save_fig(path=run_path + 'SR_LP_C_P.pdf', artists=artists)

def get_mean_sr(experiment_path, max_len, max_seeds, ref='with_init'):
    conditions = os.listdir(experiment_path)
    sr = np.zeros([max_seeds, len(conditions), max_len])
    sr.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['Eval_SR_{}'.format(i)] for i in range(NB_VALID_GOALS)]), axis=0)
            sr[i_run, i_cond, :all_sr.size] = all_sr.copy()


    sr_per_cond_stats = np.zeros([len(conditions), max_len, 3])
    sr_per_cond_stats[:, :, 0] = line(sr)
    sr_per_cond_stats[:, :, 1] = err_min(sr)
    sr_per_cond_stats[:, :, 2] = err_plus(sr)


    x_eps = np.arange(0, max_len, FREQ)# * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, max_len, FREQ)

    # compute p value wrt ref id
    p_vals = dict()
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            p_vals[i_cond] = []
            for i in x:
                p_vals[i_cond].append(ttest_ind(sr[:, i_cond, i], sr[:, ref_id, i], equal_var=False)[1])

    artists, ax = setup_figure(#xlabel='Episodes (x$10^3$)',
                               xlabel='Epochs',
                               ylabel='Success Rate',
                               xlim=[-1, x_eps[-1]],
                               ylim=[-0.02, 1.2])

    for i in range(len(conditions)):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            inds_sign = np.argwhere(np.array(p_vals[i_cond]) < ALPHA_TEST).flatten()
            if inds_sign.size > 0:
                plt.scatter(x=x_eps[inds_sign], y=np.ones([inds_sign.size]) + 0.03 + 0.06 * i_cond, marker='*', color=colors[i_cond], s=1000)
    leg = plt.legend(conditions,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=3,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 50, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    save_fig(path=SAVE_PATH + PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()


for PLOT in TO_PLOT:
    print('Plotting', PLOT)
    if PLOT == 'init_study':
        experiment_path = RESULTS_PATH + PLOT + '/'
        max_len, max_seeds = check_length_and_seeds(experiment_path=experiment_path)
        print('# epochs: {}, # seeds: {}'.format(max_len, max_seeds))
        # plot c, lp , p and sr for each run
        # plot_c_lp_p_sr(experiment_path)

        sr_per_cond_stats = get_mean_sr(experiment_path, max_len, max_seeds, ref='with_init')



