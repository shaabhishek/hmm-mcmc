from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from experiment_analysis import compute_pi, loadmodel, load_mh_uniform_transitionmatrix, load_mh_prior_transitionmatrix, \
    loaddata
from experiments import gibbs


def compute_omega(T, K, alg, model, X):
    if alg == 'gibbs':
        rw_matrix = gibbs.transition_kernel(X, model).T  # As defined in the course
    elif alg == 'mh_uniform':
        rw_matrix = load_mh_uniform_transitionmatrix(T, K)
    elif alg == 'mh_prior':
        rw_matrix = load_mh_prior_transitionmatrix(T, K)
    eigvals = np.linalg.eigvals(rw_matrix)
    order = np.argsort(eigvals)[::-1]

    w = eigvals[order]
    omega = w[1]
    assert np.abs(np.real(omega) - np.linalg.norm(omega)) < 1e-10
    omega = np.real(omega)
    # print(f"T: {T}, K: {K}, Omega: {omega:.5f}")
    return omega


def collect_results(N):
    results = defaultdict(list)
    for K in range(2, 6):
        model = loadmodel(K=K)
        for T in range(2, 10):
            if K ** T > 1200: break
            assert model.num_states == K
            results[(K, T)] = Parallel(n_jobs=8)(
                delayed(compute_omega)(T=5, K=2, alg='gibbs', model=model, X=model.sample(T=T)[1]) for i in range(N))
            # for i in range(10):
            #     _, X = model.sample(T=T)
            #     results[(K, T)].append(compute_omega(T=5, K=2, alg='gibbs', model=model, X=X))

    results_df = pd.DataFrame(results)
    return results_df


def plot_results_omega_K(results_df):
    idxs = pd.IndexSlice

    T_vals = np.stack(results_df.keys().values)[:, 1]

    for T in np.unique(T_vals):
        subset = results_df.loc[:, idxs[:, T]]
        means = subset.mean(0).droplevel(1)
        stds = subset.std(0).droplevel(1)
        plt.errorbar(means.index.to_list(), means.to_list(), stds * 2, label=f"T:{T}")
    # for (K,T), omega_vals in results_dict.items():
    #     plt.scatter(K, np.mean(omega_vals), label=f"T:{T}")
    plt.ylabel('$\omega_{\pi}$')
    plt.xlabel('K')
    plt.legend()
    # plt.show()
    plt.savefig(f"experiments/plots/omega_k_n_{len(subset)}.png")
    plt.close()

def plot_results_omega_T(results_df):
    idxs = pd.IndexSlice

    K_vals = np.array(list(results_df.keys()))[:, 0]

    for K in np.unique(K_vals):
        subset = results_df.loc[:, idxs[K, :]]
        means = subset.mean(0).droplevel(0)
        stds = subset.std(0).droplevel(0)
        plt.errorbar(means.index.to_list(), means.to_list(), stds * 2, label=f"K:{K}")
    # for (K,T), omega_vals in results_dict.items():
    #     plt.scatter(K, np.mean(omega_vals), label=f"T:{T}")
    plt.ylabel('$\omega_{\pi}$')
    plt.xlabel('T')
    plt.legend()
    # plt.show()
    plt.savefig(f"experiments/plots/omega_t_n_{len(subset)}.png")
    plt.close()


def loadresults(N):
    results_df = pd.read_csv(f'experiments/data/omega_K_T_df_n_{N}.csv', header=[0, 1], index_col=0)
    results_df.columns.names = ['K', 'T']
    return results_df


def saveresults(results_df, N):
    results_df.to_csv(f'experiments/data/omega_K_T_df_n_{N}.csv')


def run_experiments():
    N = 500
    # results = collect_results(N)
    # saveresults(results, N)
    results = loadresults(N)
    plot_results_omega_K(results)
    plot_results_omega_T(results)


if __name__ == '__main__':
    run_experiments()