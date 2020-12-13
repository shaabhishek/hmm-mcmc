import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14

from experiment_analysis import loadpi, loaddatamodel, load_mh_prior_transitionmatrix, loadomega

tmix = lambda eps,pi,omega: (np.log(1/eps) + .5 * np.log(1/np.min(pi))) / (1-omega)


def plot_results_tmix_T(tmix_dict):
    results_series = pd.Series(tmix_dict)
    results_series.index.names = ['alg', 'K', 'T']

    idxs = pd.IndexSlice

    K = 2
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = np.log10(results_series.loc[idxs[alg, K, :]])
        plt.plot(subset.index.to_list(), subset.to_list(), label=f"{alglabels[alg]}")
    plt.ylabel('$ \log_{10}$ (Tmix)')
    plt.xlabel('T')
    plt.legend()
    plt.xticks(subset.index.to_list())
    # plt.yscale('log')
    # plt.show()
    plt.savefig(f"experiments/plots/tmix_k_{K}.png")
    plt.close()

def plot_results_tmix_K(tmix_dict):
    results_series = pd.Series(tmix_dict)
    results_series.index.names = ['alg', 'K', 'T']

    idxs = pd.IndexSlice

    T = 5
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = np.log10(results_series.loc[idxs[alg, :, T]])
        plt.plot(subset.index.to_list(), subset.to_list(), label=f"{alglabels[alg]}")

    plt.ylabel('$ \log_{10}$ (Tmix)')
    # plt.ylabel('$T_{{mix}}$')
    plt.xlabel('K')
    plt.xticks(subset.index.to_list())
    plt.legend()
    # plt.yscale('log')
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
    omega_dict = loadomega()
    tmix_dict = {}
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        for K in range(2, 7):
            for T in range(2, 14):
                # if K ** T > 2000: continue  # REMOVE THIS DURING FINAL RUN
                try:
                    pi = loadpi(T=T, K=K, alg='gibbs')
                    omega = omega_dict[(alg, K, T)]
                    # (_, X), model = loaddatamodel(T, K)
                    # omega = compute_omega(T, K, alg, model, X)
                    tmix_dict[(alg, K, T)] = tmix(eps, pi, omega)
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



if __name__ == '__main__':
    alglabels = {'gibbs': 'Gibbs', 'mh_uniform': 'RWMH', 'mh_prior': 'IMH'}
    do_experiments()