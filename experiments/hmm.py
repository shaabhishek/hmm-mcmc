import numpy as np
from scipy.stats import norm, multinomial

sample_categorical = lambda p: np.nonzero(multinomial(n=1, p=p).rvs(1))[1].item()


class HMM:
    def __init__(self, num_states, transition_matrix, start_dist, means, stds):
        self.num_states = num_states
        self.transition_matrix = transition_matrix
        self.start_prob = start_dist

        self.means = means
        self.stds = stds
        # Store the emission distribution compactly as a single Gaussian with diagonal covariance since all distributions are independent
        self.emission_dists = np.array([norm(loc = self.means[z], scale = self.stds[z]) for z in range(num_states)])

    def emission_loglikelihood(self, z, x):
        # if isinstance(z, (list, np.ndarray)):
        #     return np.array([self.emission_dists[_z].logpdf(x) for _z in z])
        return self.emission_dists[z].logpdf(x)

    def emission_sample(self, z):
        return self.emission_dists[z].rvs()

    def sample(self, T):
        Z_t = sample_categorical(self.start_prob)
        X_t = self.emission_sample(Z_t)

        Z_seq, X_seq = [Z_t], [X_t]

        for t in range(1, T):
            Z_t = sample_categorical(self.transition_matrix[Z_t])
            X_t = self.emission_sample(Z_t)
            Z_seq.append(Z_t)
            X_seq.append(X_t)
        return np.array(Z_seq), np.array(X_seq)

    def sample_Z(self, T):
        Z_t = sample_categorical(self.start_prob)
        Z_seq = [Z_t]
        for t in range(1, T):
            Z_t = sample_categorical(self.transition_matrix[Z_t])
            Z_seq.append(Z_t)
        return np.array(Z_seq)

    def __repr__(self):
        out = ""
        # out += f"Vocab Size: {self.vocab_size}, Num States: {self.num_states}\n"
        out += f"Start Prob: \n{self.start_prob}\n"
        out += f"Transition Matrix: \n{self.transition_matrix}\n"
        out += f"Mean: \n{self.means}\n"
        out += f"Stds: \n{self.stds}\n"
        # out += f"Emission Probs: \n{self.emission_prob}\n"
        return out

