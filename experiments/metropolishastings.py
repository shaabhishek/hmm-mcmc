import numpy as np
import matplotlib.pyplot as plt




class MH:
    def __init__(self, model, T, N):
        self.model = model
        self.T = T
        self.N = N
        self.acceptance_probs = []

    def sample(self, x_sequence):
        N = self.N
        T = self.T

        Z_samples = np.empty((N, T), dtype=int)
        #Z_samples[0] = self.model.sample_Z(T)
        Z_samples[0] = np.random.randint(self.model.num_states,size=T)
        for n in range(1, N):
            z_proposal = np.random.randint(self.model.num_states,size=T)
            log_alpha = self.compute_log_alpha(z_proposal, Z_samples[n - 1], x_sequence, self.model)
            self.acceptance_probs.append(np.exp(min(0., log_alpha)))
            if np.random.rand() <= self.acceptance_probs[-1]:
                Z_samples[n] = z_proposal
            else:
                Z_samples[n] = Z_samples[n - 1]
        return Z_samples
    def compute_log_alpha(self,z_proposal, z_previous, x_sequence, model):
        log_emission_ratios = [model.emission_loglikelihood(z_t_prop, x_t) - model.emission_loglikelihood(z_t_prev, x_t) for
                               (z_t_prop, z_t_prev, x_t) in zip(z_proposal, z_previous, x_sequence)]
        # import pdb; pdb.set_trace()
        log_alpha = sum(log_emission_ratios)
        return log_alpha


    def plot_acceptance(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.acceptance_probs)
        plt.show()
