from itertools import product

import numpy as np
from scipy.special import softmax
from scipy.stats import multinomial

from hmm import HMM


def sample_from_markov_blanket(t, T, obs_z_sequence, obs_x_sequence, model):
    logits_z = compute_gibbs_transition_dist(t, T, obs_z_sequence, obs_x_sequence, model)
    # Convoluted way of sampling from a Cat() distribution because scipy doesn't have a direct method
    # z_t = np.nonzero(multinomial(n=1, p=softmax(logits_z)).rvs(1))[1]
    z_t = np.random.choice(np.arange(model.num_states), p=softmax(logits_z))
    return z_t


def compute_gibbs_transition_dist(t, T, obs_z_sequence, obs_x_sequence, model):
    try:
        assert obs_z_sequence.shape == obs_x_sequence.shape == (T,)

        support = np.arange(model.num_states)
        if t == 0:
            log_incoming_transition = model.start_prob[support]
        else:
            log_incoming_transition = model.transition_matrix[obs_z_sequence[t-1]][support]
        log_outgoing_transition = model.transition_matrix[:, obs_z_sequence[t+1]] if t < (T - 1) else np.zeros_like(support)  # (K,)
        log_emission_density = np.array([model.emission_loglikelihood(z, obs_x_sequence[t]) for z in support])  # (K,)
        assert log_emission_density.shape == log_incoming_transition.shape == log_outgoing_transition.shape == (
            model.num_states,)
        logits_z = log_incoming_transition + log_outgoing_transition + log_emission_density
    except:
        import pdb; pdb.set_trace()
    return logits_z


def transition_kernel(obs_x_sequence, model: HMM):
    T = len(obs_x_sequence)
    K = model.num_states
    # transition kernel size k^t * k^t
    n = K**T
    print ('transition kernel size {}*{}'.format(K**T,K**T))
    states = np.vstack(list(product(range(K),repeat = T)))
    assert len(states)==n
    rw_matrix_gibbs = np.zeros([n,n])
    for i, current_state in enumerate(states):
        for t in range(T):
            outgoing_logits = compute_gibbs_transition_dist(t, T, current_state, obs_x_sequence, model)
            outgoing_state_idxs = np.where((np.delete(states, t, axis=1) == np.delete(current_state, t)).all(axis=1))[0]
            rw_matrix_gibbs[i][outgoing_state_idxs] += softmax(outgoing_logits)/T

    return rw_matrix_gibbs


class Gibbs:
    def __init__(self, model: HMM, T: int):
        self.model = model
        self.T = T

    def sample(self, N, x_sequence, initial_sample=None):
        T = self.T

        Z_samples = np.empty((N, T), dtype=int)
        Z_samples[0] = initial_sample if initial_sample is not None else 0

        for n in range(1, N):
            Z_samples[n] = Z_samples[n - 1]
            for t in range(T):
                Z_samples[n, t] = sample_from_markov_blanket(t, T, Z_samples[n], x_sequence, self.model)

        return Z_samples


def compute_omega_gibbs(X, model):
    rw_matrix = transition_kernel(X, model).T  # As defined in the course
    # w = np.sort(np.linalg.eigvals(rw_matrix))[::-1]
    eigvals, eigvecs = np.linalg.eig(rw_matrix)
    order = np.argsort(eigvals)[::-1]
    w = eigvals[order]
    pi = eigvecs[:,order[0]]
    pi = pi/pi.sum()
    # print(eigvals)
    # print(eigvecs)
    omega = w[1]
    print(f"Eigvals: {w}, Omega: {omega:.5f}, Pi: {pi.round(1)}")


def generate_samples():
    model = HMM.from_fixed_params()
    T = 8
    Z, X = model.sample(T)
    print(model)
    print(f"X: {X}, True Z: {Z}")

    # X = X*0 + 1
    gibbs = Gibbs(model, T)

    samples = gibbs.sample(100, X)
    print(f"Last 10 samples: \n{samples[-10:]}")
    # print(np.unique(samples[:, 0], return_counts=True)[1] / len(samples))
    print(f"Mean sample: {samples.mean(0).round(1)}")
    return X, model, samples


if __name__ == '__main__':
    X, model, samples = generate_samples()
    compute_omega_gibbs(X, model)
