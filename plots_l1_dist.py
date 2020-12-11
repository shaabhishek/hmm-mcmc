import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from experiment_analysis import loadpi


def compute_l1_distance_from_samples(samples, pi_vector):
    empirical_pi_vector = np.zeros_like(pi_vector)
    for sample in samples: empirical_pi_vector[sample] += 1
    empirical_pi_vector /= empirical_pi_vector.sum()
    l1_dist = np.sum(np.abs(pi_vector - empirical_pi_vector))
    return l1_dist


def collect_results(N):
    l1_distances = {}
    for K in range(2, 7):
        for T in range(2, 14):
            try:
                pi_vector = loadpi(T=T, K=K, alg='gibbs')
            except:
                continue
            for sample_size in range(1, 11):
                samples_pi = np.random.choice(K ** T, size=(N, sample_size * 1000), p=pi_vector)
                l1_distances_KT = Parallel(n_jobs=8)(
                    delayed(compute_l1_distance_from_samples)(samples_pi[n], pi_vector) for n in range(N))
                l1_distances[(K, T, sample_size * 1000)] = l1_distances_KT
                # plt.scatter(sample_size * 1000, np.mean(l1_distances_KT))

    results_df = pd.DataFrame(l1_distances)
    return results_df


def plot_results_l1_distances(results_df):
    idxs = pd.IndexSlice

    KT_vals = set((k, t) for k, t, _ in results_df.keys().values)
    # T_vals = np.array(list(results_df.keys()))[:, 1]
    print(f'Total K-T combinations: {len(KT_vals)}')

    for (K, T) in KT_vals:
        print(f'Plotting K:{K}, T:{T}')
        subset = results_df.loc[:, idxs[K, T, :]]
        means = subset.mean(0).droplevel([0, 1])
        stds = subset.std(0).droplevel([0, 1])
        plt.errorbar(means.index.to_list(), means.to_list(), stds * 2, label=f"K:{K},T:{T}")
        plt.legend()
        plt.ylabel('$\|\hat{\pi} - \pi\|_1$')
        plt.xlabel('Number of samples')
        plt.savefig(f"experiments/plots/l1_distances_k_{K}_t_{T}.png")
        plt.close()


def loadresults(N):
    results_df = pd.read_csv(f'experiments/data/l1_distances_df_n_{N}.csv', header=[0, 1, 2], index_col=0)
    results_df.columns.names = ['K', 'T', 'sample_size']
    return results_df


def saveresults(results_df, N):
    results_df.to_csv(f'experiments/data/l1_distances_df_n_{N}.csv')


def do_experiments():
    N = 10
    # results = collect_results(N)
    # saveresults(results, N)
    l1_distances_df = loadresults(N)
    plot_results_l1_distances(l1_distances_df)


if __name__ == '__main__':
    do_experiments()
