from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from experiments.gibbs import compute_gibbs_transition_dist

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import norm

from experiment_analysis import compute_pi, loadmodel, loaddata
from experiments import gibbs
from experiments.hmm import HMM


def load_mh_prior_transitionmatrix(T,K, model:HMM, X):
    # Z, X = loaddata(T, K)
    # with open('experiments/data/hmm_k_{}.pkl'.format(K),'rb') as f:
    #     d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = model.means,scale = model.stds)
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = model.means,scale = model.stds)))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(model.transition_matrix.flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(model.start_prob[z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    print('transition kernel size {}*{}'.format(K ** T, K ** T))
    mh_prior_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_prior_transition_kernel[:,i] = np.minimum(table-table[i],0.)+log_z
        mh_prior_transition_kernel[i,i] = -np.infty
        mh_prior_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_prior_transition_kernel[:,i])))
    return np.exp(mh_prior_transition_kernel)

def load_mh_uniform_transitionmatrix(T,K, model:HMM, X):
    # Z, X = loaddata(T, K)
    # with open('experiments/data/hmm_k_{}.pkl'.format(K),'rb') as f:
    #     d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = model.means,scale = model.stds)
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = model.means,scale = model.stds)))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(model.transition_matrix.flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(model.start_prob[z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    print('transition kernel size {}*{}'.format(K ** T, K ** T))
    mh_uniform_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_uniform_transition_kernel[:,i] = np.minimum(table-table[i]+log_z-log_z[i],0)+np.log(1./n)
        mh_uniform_transition_kernel[i,i] = -np.infty
        mh_uniform_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_uniform_transition_kernel[:,i])))
    return np.exp(mh_uniform_transition_kernel)


# def compute_Wx_gibbs(rw_matrix_gibbs, obs_x_sequence, model, x):
#     T = len(obs_x_sequence)
#     K = model.num_states
#     states = np.vstack(list(product(range(K), repeat=T)))
#     y = np.zeros_like(x)
#     for i, current_state in enumerate(states):
#         for t in range(T):
#             outgoing_logits = compute_gibbs_transition_dist(t, T, current_state, obs_x_sequence, model)
#             outgoing_state_idxs = np.where((np.delete(states, t, axis=1) == np.delete(current_state, t)).all(axis=1))[0]
#             y[outgoing_state_idxs] = rw_matrix_gibbs[i][outgoing_state_idxs] * x[outgoing_state_idxs]
#     return y
# linearoperator = compute_Wx_gibbs()

def compute_omega(T, K, alg, model, X):
    if alg == 'gibbs':
        rw_matrix = gibbs.transition_kernel(X, model).T  # As defined in the course
    elif alg == 'mh_uniform':
        rw_matrix = load_mh_uniform_transitionmatrix(T, K, model, X)
    elif alg == 'mh_prior':
        rw_matrix = load_mh_prior_transitionmatrix(T, K, model, X)
    eigvals = np.linalg.eigvals(rw_matrix)
    order = np.argsort(eigvals)[::-1]

    w = eigvals[order]
    omega = w[1]
    assert np.abs(np.real(omega) - np.linalg.norm(omega)) < 1e-10
    omega = np.real(omega)
    # print(f"T: {T}, K: {K}, Omega: {omega:.5f}")
    return omega


# def collect_results_omega():
#     omega_dict = {}
#     for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
#         for K in range(2, 7):
#             for T in range(2, 14):
#                 if K**T > 10000: continue
#                 try:
#                     omega = compute_omega(T=T, K=K, alg=alg)
#                     omega_dict[(alg, K, T)] = omega
#                 except FileNotFoundError:
#                     continue
#     print(omega_dict)
#     saveomega(omega_dict)

def collect_results(N):
    omega_dict = {}
    KT_pairs = [(2,T) for T in range(2, 14)] + [(K,5) for K in range(2, 7)]
    print(KT_pairs)
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        print(f"algorithm: {alg}")
        for K,T in KT_pairs:
            if K ** T > 5000: continue
            model = loadmodel(K=K)
            omega_dict[(alg, K, T)] = Parallel(n_jobs=8)(delayed(compute_omega)(T=T, K=K, alg=alg, model=model, X=model.sample(T=T)[1]) for i in range(N))
        # for K in range(2, 6):
        #     try: model = loadmodel(K=K)
        #     except FileNotFoundError: continue
        #     assert model.num_states == K
        #     for T in range(2, 12):
        #         if K ** T > 5000: break
        #         omega_dict[(alg, K, T)] = Parallel(n_jobs=8)(delayed(compute_omega)(T=T, K=K, alg=alg, model=model, X=model.sample(T=T)[1]) for i in range(N))
            # results[(K, T)] = Parallel(n_jobs=8)(
            #     delayed(compute_omega)(T=5, K=2, alg='gibbs', model=model, X=model.sample(T=T)[1]) for i in range(N))
            # for i in range(10):
            #     _, X = model.sample(T=T)
            #     results[(K, T)].append(compute_omega(T=5, K=2, alg='gibbs', model=model, X=X))

    results_df = pd.DataFrame(omega_dict)
    return results_df


def plot_results_omega_K(results_df):
    T=5
    idxs = pd.IndexSlice

    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = results_df.loc[:, idxs[alg, :, T]]
        means = 1-subset.mean(0).droplevel([0,2])
        stds = subset.std(0).droplevel([0,2])
        # plt.plot(subset.index.to_list(), subset.to_list(), label=f"{alg}")
        plt.errorbar(means.index.to_list(), means.to_list(), yerr=stds/np.sqrt(len(subset)), capsize=4, label=f"{alglabels[alg]}")

    plt.ylabel('$1 - \omega_{\pi}$')
    plt.xlabel('Hidden State Dimension (K)')
    plt.xticks(means.index.to_list())
    plt.legend()
    plt.yticks(np.linspace(0, .6, 7))
    plt.ylim(0, .6)
    plt.title('T=5')
    # plt.show()
    plt.savefig(f"experiments/plots/omega_t_5_vary_k_vary_alg_n_{len(subset)}.png")
    plt.close()

def plot_results_omega_T(results_df):
    idxs = pd.IndexSlice

    K=2
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        subset = results_df.loc[:, idxs[alg, K, :]]
        means = 1-subset.mean(0).droplevel([0,1])
        stds = subset.std(0).droplevel([0,1])
        plt.errorbar(means.index.to_list(), means.to_list(), yerr=stds/np.sqrt(len(subset)), capsize=4, label=f"{alglabels[alg]}")

    plt.ylabel('$1 - \omega_{\pi}$')
    plt.xlabel('Trajectory Length (T)')
    plt.xticks(means.index.to_list())
    plt.yticks(np.linspace(0, .6, 7))
    plt.ylim(0, .6)
    plt.title('K=2')
    plt.legend()
    # plt.show()
    plt.savefig(f"experiments/plots/omega_k_2_vary_t_vary_alg_n_{len(subset)}.png")
    plt.close()


    # alg = 'gibbs'
    # K_vals = np.array(list(results_df.keys()))[:, 1]
    # for K in np.unique(K_vals):
    #     subset = results_df.loc[:, idxs[alg, K, :]]
    #     means = subset.mean(0).droplevel([0, 1])
    #     stds = subset.std(0).droplevel([0, 1])
    #     plt.errorbar(means.index.to_list(), means.to_list(), stds, label=f"K:{K}")
    #
    # plt.ylabel('$\omega_{\pi}$')
    # plt.xlabel('T')
    # plt.ylim(.5, 1.1)
    # plt.legend()
    # # plt.show()
    # plt.savefig(f"experiments/plots/omega_gibbs_vary_t_vary_k_n_{len(subset)}.png")
    # plt.close()


def loadresults(N):
    # results_df = pd.read_csv(f'experiments/data/omega_K_T_df_n_{N}.csv', header=[0, 1, 2], index_col=0)
    # results_df.columns.names = ['alg', 'K', 'T']

    from compute_tmix import load_pi_omega
    _, omega_dict = load_pi_omega()

    results_df = pd.DataFrame(omega_dict)
    results_df.columns.names = ['alg', 'K', 'T']

    return results_df


def saveresults(results_df, N):
    results_df.to_csv(f'experiments/data/omega_K_T_df_n_{N}.csv')


def run_experiments():
    N = 24
    # results = collect_results(N)
    # saveresults(results, N)
    results = loadresults(N)
    plot_results_omega_K(results)
    plot_results_omega_T(results)


if __name__ == '__main__':
    alglabels = {'gibbs': 'Gibbs', 'mh_uniform': 'RWMH', 'mh_prior': 'IMH'}
    run_experiments()