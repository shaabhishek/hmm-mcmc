import numpy as np
import ipdb



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
        Z_samples[0] = self.model.sample_Z(T)
        acc_t = 0.
        for n in range(1, N):
            z_proposal = self.model.sample_Z(T)
            log_alpha = self.compute_log_alpha(z_proposal, Z_samples[n - 1], x_sequence, self.model)
            self.acceptance_probs.append(log_alpha)
            if np.log(np.random.rand()) <= log_alpha:
                Z_samples[n] = z_proposal
                acc_t += 1
            else:
                Z_samples[n] = Z_samples[n - 1]

        print ('Acc rate',acc_t*1./N)
        return Z_samples,self.acceptance_probs

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


class MH_Uniform:
    def __init__(self, model, T, N):
        self.model = model
        self.T = T
        self.N = N
        self.acceptance_probs = []
        self.loglik_tab = np.log(model.transition_matrix.flatten())
    def sample(self, x_sequence):
        N = self.N
        T = self.T

        Z_samples = np.empty((N, T), dtype=int)
        Z_samples[0] = np.random.randint(self.model.num_states,size=T)
        acc_t = 0
        for n in range(1, N):
            z_proposal = np.random.randint(self.model.num_states,size=T)
            log_alpha = self.compute_log_alpha(z_proposal, Z_samples[n - 1], x_sequence, self.model)
            self.acceptance_probs.append(log_alpha)
            if np.log(np.random.rand()) <= log_alpha:
                Z_samples[n] = z_proposal
                acc_t += 1
            else:
                Z_samples[n] = Z_samples[n - 1]
        print ('Acc rate',acc_t*1./N)
        return Z_samples,self.acceptance_probs

    def compute_log_alpha(self,z_proposal, z_previous, x_sequence, model):
        log_emission_ratios = [model.emission_loglikelihood(z_t_prop, x_t) - model.emission_loglikelihood(z_t_prev, x_t) for
                               (z_t_prop, z_t_prev, x_t) in zip(z_proposal, z_previous, x_sequence)]
        log_z_diff = np.log(model.start_prob[z_proposal[0]])-np.log(model.start_prob[z_previous[0]])
        f = lambda t: t[1]*model.num_states+t[0]
        log_old_z = np.log(model.start_prob[z_previous[0]]) + np.sum(self.loglik_tab[list(map(f,zip(z_previous[:-1],z_previous[1:])))])
        log_new_z = np.log(model.start_prob[z_proposal[0]]) + np.sum(self.loglik_tab[list(map(f,zip(z_proposal[:-1],z_proposal[1:])))])
        log_alpha = sum(log_emission_ratios)+log_new_z-log_old_z
        return log_alpha


    def plot_acceptance(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.acceptance_probs)
        plt.show()
