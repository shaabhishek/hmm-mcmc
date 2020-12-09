import pickle
import numpy as np

import gibbs as gibbs1
import experiments.gibbs as gibbs2
import hmm as hmm1
import experiments.hmm as hmm2


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

def loadtransitionmatrix(T, K, modelversion=2):
    Z, X = loaddata(T, K)
    model = loadmodel(K, modelversion)
    return gibbs1.transition_kernel(X, model) if modelversion == 1 else gibbs2.transition_kernel(X, model)
    # return gibbs2.transition_kernel(X, model)

def test_two_versions():
    for T in  range(2, 10):
        try:
            assert np.allclose(loadtransitionmatrix(T, 2, 1), loadtransitionmatrix(T, 2, 2))
        except:
            import ipdb; ipdb.set_trace()


def savepi(T, K, alg, pi_vector, ):
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

def compute_pi(T, K):
    (_,X), model = loaddatamodel(T, K)
    rw_matrix = gibbs2.transition_kernel(X, model).T  # As defined in the course
    eigvals, eigvecs = np.linalg.eig(rw_matrix)
    order = np.argsort(eigvals)[::-1]
    pi = eigvecs[:, order[0]]
    pi = pi / pi.sum()

    w = eigvals[order]
    omega = w[1]
    print(f"T: {T}, K: {K}, Omega: {omega:.5f}, Pi: {pi.round(1)}")
    return pi



if __name__ == '__main__':
    for t in range(2, 14):
        # pi = compute_pi(T=t, K=2)
        # savepi(t, K=2, alg='gibbs', pi_vector=pi)
        loadpi(t, K=2, alg='gibbs')
