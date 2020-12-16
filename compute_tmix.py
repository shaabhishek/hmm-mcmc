import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14

from experiment_analysis import loadpi, loaddatamodel, load_mh_prior_transitionmatrix, loadomega

tmix = lambda eps,pi,omega: (np.log(1/eps) + .5 * np.log(1/np.min(pi))) / (1-omega)


def plot_results_tmix_T(tmix_dict):
    results_df = pd.DataFrame(tmix_dict)
    results_df.columns.names = ['alg', 'K', 'T']

    idxs = pd.IndexSlice

    K = 2
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = np.log10(results_df.loc[:, idxs[alg, K, :]])
        means = subset.mean(0).droplevel([0, 1])
        stds = subset.std(0).droplevel([0, 1])
        plt.errorbar(means.index.to_list(), means.to_list(), yerr=stds / np.sqrt(len(subset)), capsize=4,
                     label=f"{alglabels[alg]}")
        # subset = np.log10(results_df.loc[idxs[alg, K, :]])
        # plt.plot(subset.index.to_list(), subset.to_list(), label=f"{alglabels[alg]}")
    plt.ylabel('$ \log_{10}$ (Tmix)')
    plt.xlabel('Trajectory Length (T)')
    plt.ylim(1, 5)
    plt.xticks(means.index.to_list())
    plt.yticks(np.arange(1, 6))
    plt.legend(loc='upper left')
    plt.title('K=2')
    # plt.show()
    plt.savefig(f"experiments/plots/tmix_k_{K}.png")
    plt.close()

def plot_results_tmix_K(tmix_dict):
    results_df = pd.DataFrame(tmix_dict)
    results_df.columns.names = ['alg', 'K', 'T']

    idxs = pd.IndexSlice

    T = 5
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = np.log10(results_df.loc[:, idxs[alg, :, T]])
        means = subset.mean(0).droplevel([0, 2])
        stds = subset.std(0).droplevel([0, 2])
        plt.errorbar(means.index.to_list(), means.to_list(), yerr=stds / np.sqrt(len(subset)), capsize=4,
                     label=f"{alglabels[alg]}")
        # subset = np.log10(results_df.loc[idxs[alg, :, T]])
        # plt.plot(subset.index.to_list(), subset.to_list(), label=f"{alglabels[alg]}")

    plt.ylabel('$ \log_{10}$ (Tmix)')
    # plt.ylabel('$T_{{mix}}$')
    plt.xlabel('Hidden State Dimension (K)')
    plt.ylim(1, 5)
    plt.xticks(means.index.to_list())
    plt.yticks(np.arange(1, 6))
    plt.legend(loc='upper left')
    plt.title('T=5')
    # plt.show()
    plt.savefig(f"experiments/plots/tmix_T_{T}.png")
    plt.close()



def do_experiments():
    # tmix_dict = make_tmix()
    # savetmix(tmix_dict)
    tmix_dict = loadtmix()
    plot_results_tmix_T(tmix_dict)
    plot_results_tmix_K(tmix_dict)


def make_tmix():
    alg = 'gibbs'
    eps = 0.01
    pi_dict, omega_dict = load_pi_omega()
    tmix_dict = {}
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        for K in range(2, 7):
            for T in range(2, 14):
                # if K ** T > 2000: continue  # REMOVE THIS DURING FINAL RUN
                try:
                    pi_vals = pi_dict[(K,T)] #array of shape (N, K^T)
                    omega_vals = omega_dict[(alg, K, T)] #list of length N
                    # (_, X), model = loaddatamodel(T, K)
                    # omega = compute_omega(T, K, alg, model, X)
                    tmix_dict[(alg, K, T)] = [tmix(eps, pi, omega) for pi,omega in zip(pi_vals, omega_vals)]
                    print(f"T_mix for {alg} K:{K}, T:{T}: {tmix_dict[(alg, K, T)]}")
                except:
                    continue
    return tmix_dict

def savetmix(tmix_dict):
    assert isinstance(tmix_dict, dict)
    filename = f'experiments/data/tmix_dict_for_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(tmix_dict, f)
    print(f"Saved to {filename}")

def loadtmix():
    filename = f'experiments/data/tmix_dict_for_data.pkl'
    with open(filename, 'rb') as f:
        tmix_dict = pickle.load(f)

    assert isinstance(tmix_dict, dict)
    print(f"Loaded Omega Dict from {filename}. Keys (Alg, K, T): {tmix_dict.keys()}")
    return tmix_dict


def load_pi_omega():
    foldername = Path(f'experiments/results/mix_time')
    filenames = {(K,T): foldername / f'mix_time_analysis_K_{K}_T_{T}.pkl' for K in range(2, 7) for T in range(2, 14) if (foldername / f'mix_time_analysis_K_{K}_T_{T}.pkl').exists()}

    omega_dict = {}
    pi_dict = {}
    for (K,T), filename in filenames.items():
        with open(filename, 'rb') as f:
            _data_KT = pd.DataFrame(pickle.load(f)).T
            pi_dict[(K, T)] = np.stack(_data_KT['pi'].to_list(), 0)
            for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
                omega_dict[(alg, K, T)] = _data_KT[f'w_{alg}'].to_list()

    assert isinstance(omega_dict, dict)
    assert isinstance(pi_dict, dict)
    print(f"Loaded Pi/Omega values from Data from {foldername}. Keys (Alg, K, T): {omega_dict.keys()}")
    return pi_dict, omega_dict


if __name__ == '__main__':
    alglabels = {'gibbs': 'Gibbs', 'mh_uniform': 'RWMH', 'mh_prior': 'IMH'}
    do_experiments()