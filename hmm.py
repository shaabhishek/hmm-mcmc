import numpy as np
from hmmlearn import hmm as hmm
from scipy.special import log_softmax, softmax
from scipy.stats import dirichlet, multivariate_normal, multinomial

sample_categorical = lambda logits: np.nonzero(multinomial(n=1, p=softmax(logits)).rvs(1))[1].item()

np.set_printoptions(precision=2)


class HMM:
    def __init__(self, num_states, transition_matrix, start_dist, means, covars):
        self.num_states = num_states
        self.transition_matrix = transition_matrix
        self.start_prob = start_dist

        self.means = means
        self.covars = covars
        # Store the emission distribution compactly as a single Gaussian with diagonal covariance since all distributions are independent
        self.emission_dists = [multivariate_normal(self.means[z], self.covars[z]) for z in range(num_states)]

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
        out += f"Start Prob: \n{softmax(self.start_prob)}\n"
        out += f"Transition Matrix: \n{softmax(self.transition_matrix, axis=-1)}\n"
        out += f"Mean: \n{self.means}\n"
        # import pdb; pdb.set_trace()
        out += f"Covar: \n{self.covars}\n"
        # out += f"Emission Probs: \n{self.emission_prob}\n"
        return out

    @classmethod
    def from_fixed_params(cls):
        num_states = 2
        # transition_matrix = log_softmax(np.eye(num_states), axis=-1)
        # start_dist = log_softmax(np.ones(num_states), axis=-1)
        transition_matrix = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
        start_dist = np.log(np.array([0.3, 0.7]))

        means = np.array([-1., 1.])
        covars = .5 * np.ones(num_states)
        self = cls(num_states, transition_matrix, start_dist, means, covars)

        return self


# class HMM_old:
#     def __init__(self):
#         self.num_states = 2
#         self.transition_matrix = log_softmax(np.random.rand(self.num_states, self.num_states), axis=-1)
#         self.start_prob = log_softmax(np.random.rand(self.num_states), axis=-1)
#
#         # self.vocab_size= 3
#         # self.emission_prob = softmax(np.random.rand(self.num_states, self.vocab_size), axis=-1)
#         self.means = np.random.rand(self.num_states, 1)
#         self.covars = np.random.rand(self.num_states)
#
#         self.init_hmm(self.num_states, {'means': self.means, 'covars': self.covars}, self.transition_matrix, self.start_prob)
#
#     def init_hmm(self, num_states, emission_prob, transition_matrix, start_prob):
#         # self._hmm = hmm.MultinomialHMM(n_components=num_states, init_params='', n_iter=100)
#         # self._hmm.emissionprob_ = emission_prob
#         self._hmm = hmm.GaussianHMM(n_components=num_states, n_iter=100, covariance_type='spherical', tol=.001, transmat_prior=.01)
#         self._hmm.means_ = emission_prob['means']
#         self._hmm.covars_ = emission_prob['covars']
#         self._hmm.transmat_ = transition_matrix
#         self._hmm.startprob_ = start_prob
#         self._hmm._check()
#         # self._hmm.monitor_.verbose = True
#
#     def emission_loglikelihood(self, z, x):
#         return multivariate_normal(self.means[z], self.covars[z]).logpdf(x)
#
#     def sample_Z(self, N=1, T=4):
#         return self.sample(N, T)[1]
#
#     def sample(self, N, T):
#         X_seqs, Z_seqs = [], []
#         for _ in range(N):
#             X, Z = self._hmm.sample(T)
#             X_seqs.append(X)
#             Z_seqs.append(Z)
#         return np.stack(X_seqs, axis=0), np.stack(Z_seqs, axis=0)
#
#     def __repr__(self):
#         out = ""
#         # out += f"Vocab Size: {self.vocab_size}, Num States: {self.num_states}\n"
#         out += f"Start Prob: \n{self.start_prob}\n"
#         out += f"Transition Matrix: \n{self.transition_matrix}\n"
#         out += f"Mean: \n{self._hmm.means_}\n"
#         # import pdb; pdb.set_trace()
#         out += f"Covar: \n{self._hmm.covars_}\n"
#         # out += f"Emission Probs: \n{self.emission_prob}\n"
#         return out
#
#     @classmethod
#     def from_fixed_params(cls):
#         self = cls()
#         self.transition_matrix = dirichlet(.8*np.ones(self.num_states)).rvs(self.num_states)
#         self.start_prob = dirichlet(.8*np.ones(self.num_states)).rvs().flatten()
#         # self.start_prob = softmax(np.random.rand(self.num_states), axis=-1)
#         # self.transition_matrix = softmax(np.random.rand(self.num_states, self.num_states), axis=-1)
#
#         # self.emission_prob = dirichlet(np.random.rand(self.vocab_size)).rvs(self.num_states)
#         self.means = 20*np.random.randn(self.num_states, 1)
#         self.covars = 5*np.ones(self.num_states)
#
#         self.init_hmm(self.num_states, {'means': self.means, 'covars': self.covars}, self.transition_matrix, self.start_prob)
#
#
#         # self._hmm = hmm.MultinomialHMM(n_components=self.num_states, init_params='')
#         # self._hmm.emissionprob_ = self.emission_prob
#         # self._hmm.transmat_ = self.transition_matrix
#         # self._hmm.startprob_ = self.start_prob
#         return self
#
#     @classmethod
#     def from_initialized_hmm(cls, initialized_hmm: hmm.MultinomialHMM):
#         instance = cls()
#         instance.num_states = initialized_hmm.n_components
#         instance.vocab_size = initialized_hmm.n_features
#         instance._hmm = initialized_hmm
#         instance.emission_prob = initialized_hmm.emissionprob_
#         instance.transition_matrix = initialized_hmm.transmat_
#         instance.start_prob = initialized_hmm.startprob_
#         return instance
#
#     def fit(self, X):
#         N, T, _ = X.shape
#         X_flattened = X.reshape(-1,1)
#         lengths = np.ones(N, dtype=int)*T
#         assert np.all(X_flattened[:T] == X[0])
#         assert sum(lengths) == N*T
#         self._hmm.fit(X_flattened, lengths=lengths)
#         return self
#
#     def _compute_sufficient_stats(self, Z_seqs):
#         from itertools import tee
#         from collections import defaultdict
#
#         N, T = Z_seqs.shape
#
#         sufficient_stats_pair_counts = defaultdict(int)
#         sufficient_stats_pair_counts_matrix = np.zeros_like(self.transition_matrix)
#         sufficient_stats_start_counts = (Z_seqs[:, 0].reshape(-1, 1) == np.arange(self.num_states).reshape(1, -1)).sum(0)
#
#         for Z in Z_seqs:
#             zt, zt_prime = tee(Z)
#             next(zt_prime, None)
#             for pair in zip(zt, zt_prime):
#                 sufficient_stats_pair_counts_matrix[pair] += 1
#         trans_mat_estimate = sufficient_stats_pair_counts_matrix / sufficient_stats_pair_counts_matrix.sum(1, keepdims=True)
#         start_prob_estimate = sufficient_stats_start_counts / sufficient_stats_start_counts.sum()
#         return start_prob_estimate, trans_mat_estimate
#
#     def log_joint(self, Z, X):
#         assert len(Z.shape) == 1
#         assert len(X.shape) == 2
#         emission_distributions = multivariate_normal(self._hmm.means_, self._hmm.covars_)
#
#         log_joint = np.log(Z[0]) + multivariate_normal
#         for t in range(1, len(Z)):
#             log_joint += np.log(self._hmm.transmat_[Z[t-1], Z[t]])
#             log_joint += multivariate_normal.logpdf(X[t])
#
# def sanity_check(model, X, Z):
#     N, T, _ = X.shape
#     test_model = HMM()
#     test_model.fit(X)
#
#     print(test_model._hmm.monitor_)
#     print(test_model)
#     return


if __name__ == '__main__':
    model = HMM.from_fixed_params()
    print(model)
    Z, X = model.sample(T=30)
    print(X.shape, Z.shape)
    print(X, Z)
