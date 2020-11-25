import numpy as np

from hmm import HMM

np.set_printoptions(precision=2)

def compute_log_alpha(z_proposal, z_previous, x_sequence, model: HMM):
    log_emission_ratios = [model.emission_loglikelihood(z_t_prop, x_t) - model.emission_loglikelihood(z_t_prev, x_t) for (z_t_prop, z_t_prev, x_t) in zip(z_proposal, z_previous, x_sequence)]
    # import pdb; pdb.set_trace()
    log_alpha = sum(log_emission_ratios)
    return log_alpha


class MH:
    def __init__(self, model: HMM, T: int):
        self.model = model
        self.T = T

        self.acceptance_probs = []

    def sample(self, x_sequence):
        N = 50
        T = self.T

        Z_samples = np.empty((N, T), dtype=int)
        Z_samples[0] = self.model.sample_Z(T)

        for n in range(1, N):
            z_proposal = self.model.sample_Z(T)
            log_alpha = compute_log_alpha(z_proposal, Z_samples[n-1], x_sequence, self.model)
            self.acceptance_probs.append(np.exp(min(0., log_alpha)))
            if np.random.rand() <= self.acceptance_probs[-1]:
                Z_samples[n] = z_proposal
            else:
                Z_samples[n] = Z_samples[n-1]
        return Z_samples

if __name__ == '__main__':
    model = HMM.from_fixed_params()
    T = 10
    Z, X = model.sample(T)
    mh = MH(model, T)
    print(model)
    print(X, Z)
    print(mh.sample(X)[-10:])
    print(mh.acceptance_probs)
