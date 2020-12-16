import pickle
import numpy as np

import experiments.gibbs as gibbs2
import experiments.hmm as hmm2
from scipy.special import logsumexp
from scipy.stats import norm
from itertools import product
def loaddata(T, K):
    with open('experiments/data/data_t_{}_k_{}.pkl'.format(T, K), 'rb') as f:
        Z, X = pickle.load(f)
    return Z, X

def loadmodel(K, modelversion=2):
    with open('experiments/data/hmm_k_{}.pkl'.format(K), 'rb') as f:
        d = pickle.load(f)

    if modelversion == 1:
        # If the model requires logprobs
        d['transition_matrix'] = np.log(d['transition_matrix'])
        d['start_prob'] = np.log(d['start_prob'])
        return hmm1.HMM(d['num_states'],d['transition_matrix'],d['start_prob'],d['means'],d['stds'])
    elif modelversion == 2:
        return hmm2.HMM(d['num_states'],d['transition_matrix'],d['start_prob'],d['means'],d['stds'])

def loadsampler(T, N, hmm_model):
    sampler = gibbs2.Gibbs(hmm_model, T, N)
    return sampler

def loaddatamodel(T, K):
    Z, X = loaddata(T, K)
    model = loadmodel(K)
    return (Z, X), model

def load_gibbs_transitionmatrix(T, K, modelversion=2):
    Z, X = loaddata(T, K)
    model = loadmodel(K, modelversion)
    return gibbs1.transition_kernel(X, model) if modelversion == 1 else gibbs2.transition_kernel(X, model)
    # return gibbs2.transition_kernel(X, model)


def load_mh_prior_transitionmatrix(T,K):
    Z, X = loaddata(T, K)
    with open('experiments/data/hmm_k_{}.pkl'.format(K),'rb') as f:
        d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = d['means'],scale = d['stds'])
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = d['means'],scale = d['stds'])))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(d['transition_matrix'].flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(d['start_prob'][z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    mh_prior_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_prior_transition_kernel[:,i] = np.minimum(table-table[i],0.)+log_z
        mh_prior_transition_kernel[i,i] = -np.infty
        mh_prior_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_prior_transition_kernel[:,i])))
    return np.exp(mh_prior_transition_kernel) 

def load_mh_uniform_transitionmatrix(T,K):
    Z, X = loaddata(T, K)
    with open('experiments/data/hmm_k_{}.pkl'.format(K),'rb') as f:
        d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = d['means'],scale = d['stds'])
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = d['means'],scale = d['stds'])))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(d['transition_matrix'].flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(d['start_prob'][z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    mh_uniform_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_uniform_transition_kernel[:,i] = np.minimum(table-table[i]+log_z-log_z[i],0)+np.log(1./n)
        mh_uniform_transition_kernel[i,i] = -np.infty
        mh_uniform_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_uniform_transition_kernel[:,i])))
    return np.exp(mh_uniform_transition_kernel)

def test_two_versions():
    for T in  range(2, 10):
        try:
            assert np.allclose(loadtransitionmatrix(T, 2, 1), loadtransitionmatrix(T, 2, 2))
        except:
            import ipdb; ipdb.set_trace()


def savepi(T, K, alg, pi_vector):
    assert pi_vector.shape == (K**T,)
    filename = f'experiments/data/pi_{alg}_k_{K}_t_{T}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(list(pi_vector), f)

    print(f"Saved to {filename}")

def loadpi(T, K, alg,):
    filename = f'experiments/data/pi_{alg}_k_{K}_t_{T}.pkl'
    with open(filename, 'rb') as f:
        pi_vector = pickle.load(f)
    pi_vector = np.array(pi_vector)
    assert pi_vector.shape == (K**T,)

    print(f"Loaded Pi Vector from {filename}. Shape: {pi_vector.shape}")
    return pi_vector

def compute_pi(T, K,alg):
    (_,X), model = loaddatamodel(T, K)
    if alg=='gibbs':
        rw_matrix = gibbs2.transition_kernel(X, model).T  # As defined in the course
    elif alg=='mh_uniform':
        rw_matrix = load_mh_uniform_transitionmatrix(T,K)
    elif alg=='mh_prior':
        rw_matrix = load_mh_prior_transitionmatrix(T,K)
    eigvals, eigvecs = np.linalg.eig(rw_matrix)
    order = np.argsort(eigvals)[::-1]
    pi = eigvecs[:, order[0]]
    real = np.real(pi)
    
    assert np.abs(np.linalg.norm(real)-np.linalg.norm(pi))<1e-10
    pi = np.copy(real)
    pi = pi / pi.sum()

    w = eigvals[order]
    omega = w[1]
    assert np.abs(np.real(omega)-np.linalg.norm(omega))<1e-10
    omega = np.real(omega)
    print(f"T: {T}, K: {K}, Omega: {omega:.5f}, Pi: {pi}")
    return pi


def compute_omega(T, K, alg):
    (_, X), model = loaddatamodel(T, K)
    if alg == 'gibbs':
        rw_matrix = gibbs2.transition_kernel(X, model).T  # As defined in the course
    elif alg == 'mh_uniform':
        rw_matrix = load_mh_uniform_transitionmatrix(T, K)
    elif alg == 'mh_prior':
        rw_matrix = load_mh_prior_transitionmatrix(T, K)
    else: raise
    eigvals = np.linalg.eigvals(rw_matrix)
    order = np.argsort(eigvals)[::-1]

    w = eigvals[order]
    omega = w[1]
    assert np.abs(np.real(omega) - np.linalg.norm(omega)) < 1e-10
    omega = np.real(omega)
    print(f"T: {T}, K: {K}, Omega: {omega:.5f}")
    return omega

def collect_results_pi():
    for K in range(2, 7):
        for T in range(2, 14):
            if K**T > 10000: continue
            try:
                pi = compute_pi(T=T, K=K, alg='gibbs')
                savepi(T=T, K=K, alg='gibbs', pi_vector=pi)
            except FileNotFoundError:
                continue

def collect_results_omega():
    omega_dict = {}
    for alg in ['gibbs', 'mh_uniform', 'mh_prior']:
        for K in range(2, 7):
            for T in range(2, 14):
                if K**T > 10000: continue
                try:
                    omega = compute_omega(T=T, K=K, alg=alg)
                    omega_dict[(alg, K, T)] = omega
                except FileNotFoundError:
                    continue
    print(omega_dict)
    saveomega(omega_dict)

def saveomega(omega_dict):
    assert isinstance(omega_dict, dict)
    filename = f'experiments/data/omega_dict_for_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(omega_dict, f)
    print(f"Saved to {filename}")

def loadomega():
    filename = f'experiments/data/omega_dict_for_data.pkl'
    with open(filename, 'rb') as f:
        omega_dict = pickle.load(f)

    assert isinstance(omega_dict, dict)
    print(f"Loaded Omega Dict from {filename}. Keys (Alg, K, T): {omega_dict.keys()}")
    return omega_dict


def do_experiments():
    # collect_results_pi()
    collect_results_omega()

if __name__ == '__main__':
    do_experiments()
    # for t in range(2, 14):
    #     pi = compute_pi(T=t, K=2,alg='mh_prior')
        # savepi(t, K=2, alg='gibbs', pi_vector=pi)

        #loadpi(t, K=2, alg='gibbs')
    # for k in range(3, 7):
        #pi = compute_pi(5, K=k)
        #savepi(5, K=k, alg='gibbs', pi_vector=pi)
        # loadpi(5, K=k, alg='gibbs')
