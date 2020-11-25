import numpy as np
from scipy.special import softmax
from scipy.stats import multinomial

from hmm import HMM


def sample_from_markov_blanket(z_sequence, x_sequence, t, T, model):
    assert z_sequence.shape == x_sequence.shape == (T, )
    support = np.arange(model.num_states)
    # import pdb; pdb.set_trace()
    log_incoming_transition = model.transition_matrix[z_sequence[t]][support] if t > 0 else np.zeros_like(support)  # (K,)
    log_outgoing_transition = model.transition_matrix[:, z_sequence[t]] if t < T else np.zeros_like(support)  # (K,)
    log_emission_density = np.array([model.emission_loglikelihood(z, x_sequence[t]) for z in support]) # (K,)
    assert log_emission_density.shape == log_incoming_transition.shape == log_outgoing_transition.shape == (
    model.num_states,)
    logits_z = log_incoming_transition + log_outgoing_transition + log_emission_density
    # Convoluted way of sampling from a Cat() distribution because scipy doesn't have a direct method
    z_t = np.nonzero(multinomial(n=1, p=softmax(logits_z)).rvs(1))[1]
    return z_t


class Gibbs:
    def __init__(self, model: HMM, T: int):
        self.model = model
        self.T = T

    def sample(self, x_sequence):
        N = 1000
        T = self.T

        Z_samples = np.empty((N, T), dtype=int)
        Z_samples[0] = 0

        for n in range(1, N):
            Z_samples[n] = Z_samples[n - 1]
            for t in range(T):
                Z_samples[n, t] = sample_from_markov_blanket(Z_samples[n], x_sequence, t, T, self.model)

        return Z_samples


if __name__ == '__main__':
    model = HMM.from_fixed_params()
    T = 6
    Z, X = model.sample(T)
    gibbs = Gibbs(model, T)
    print(model)
    print(X, Z)
    samples = gibbs.sample(X)
    print(samples[-10:])
    # print(np.unique(samples[:,0], return_counts=True)[1]/ len(samples) )
    print(samples.mean(0).round())