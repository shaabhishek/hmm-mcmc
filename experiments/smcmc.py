import numpy as np
from joblib import Parallel, delayed
from scipy.special import softmax

from gibbs import compute_gibbs_transition_dist, Gibbs
from hmm import HMM


class SMCMC:
    def __init__(self, model: HMM, T: int):
        self.model = model
        self.T = T
        self.L = 8

    def jump(self, current_z_sequence, x_sequence):
        (T,) = x_sequence.shape
        assert current_z_sequence.shape == (T - 1,)
        # Pad the old point to make space for $nu_t$
        padded_z_sequence = np.concatenate([current_z_sequence, [0]])
        logits_z = compute_gibbs_transition_dist(T-1, T, padded_z_sequence, x_sequence, self.model)
        padded_z_sequence[T-1] = np.random.choice(np.arange(self.model.num_states), p=softmax(logits_z))
        return padded_z_sequence

    def compute_autocorrelation(self, sample_1, sample_2, dim_sample):
        assert sample_1.shape == sample_2.shape == (self.L, dim_sample)
        std_noise = 0.001
        sample_1 = sample_1 + std_noise*np.random.randn(self.L, dim_sample)
        sample_2 = sample_1 + std_noise*np.random.randn(self.L, dim_sample)

        sample_1_means = sample_1.mean(0, keepdims=True)
        sample_2_means = sample_2.mean(0, keepdims=True)

        sample_1_deviations = sample_1 - sample_1_means
        sample_2_deviations = sample_2 - sample_2_means

        sample_1_var = (sample_1_deviations**2).sum(0, keepdims=True)
        sample_2_var = (sample_2_deviations**2).sum(0, keepdims=True)

        sample_1_2_covar = (sample_1_deviations * sample_2_deviations).sum(0)

        # print(f"Sample means: {sample_1_means}, {sample_2_means}")
        # print(f"Sample vars: {sample_1_var}, {sample_2_var}")

        corr = sample_1_2_covar / np.sqrt(sample_1_var * sample_2_var)
        corr_ = np.array([np.corrcoef(sample_1[:,p], sample_2[:,p])[0,1] for p in range(dim_sample)])
        assert np.allclose(corr, corr_)
        return corr.max()



    def sample(self, N, x_sequence, initial_sample=None):
        T, L = self.T, self.L
        eps = 1e-4
        N_max = N
        batch_size = N // 5

        """
        TODO:
        1. modify gibbs to take a starting point - done
        2. implement jumping function that returns an expanded point after taking the data and previous point - done
        3. compute autocorrelation
        4. implement L parallel chains - Done
        """

        t_start = T//2
        samples = np.zeros((L, N, t_start-1), dtype=int)


        for t in range(t_start, T+1):
            # assert samples.shape == (L, N, t-1)
            x_subsequence = x_sequence[:t]
            # initial_z_sample = self.jump(samples[-1], x_subsequence)
            initial_z_sample = np.stack(Parallel(n_jobs=1)(delayed(self.jump)(samples[l][-1], x_subsequence) for l in range(L)), axis=0)
            assert initial_z_sample.shape == (L, t)
            samples = np.zeros((L, 1, t), dtype=int)
            samples[:, 0] = initial_z_sample
            # samples = []
            # while

            rho = 1
            n_t = 0
            while rho > 1 - eps:
            # for n in range(0, N, batch_size):
                gibbs = Gibbs(self.model, t)
                # samples = gibbs.sample(10, x_subsequence, initial_sample=initial_z_sample)
                # samples[:, n_t:n_t+batch_size, ] = np.stack(Parallel(n_jobs=4)(delayed(gibbs.sample)(batch_size, x_subsequence, initial_sample=initial_z_sample[l]) for l in range(L)), axis=0)
                samples_batch = np.stack(Parallel(n_jobs=4)(delayed(gibbs.sample)(batch_size, x_subsequence, initial_sample=initial_z_sample[l]) for l in range(L)), axis=0)
                samples = np.concatenate([samples, samples_batch], axis=1)
                rho = self.compute_autocorrelation(initial_z_sample, samples[:, -1], dim_sample=t)
                print(f"t: {t}, rho: {rho}, sample shape: {samples.shape}, last sample: {samples[-1, -1]}")
                n_t += batch_size
                if n_t >= N_max: break
            print(f"n_t: {n_t}, number of samples: {samples.shape}")

        return samples


def generate_samples():
    model = HMM.from_fixed_params()
    T = 10
    Z, X = model.sample(T)
    print(model)
    print(f"X: {X}, True Z: {Z}")

    smcmc = SMCMC(model, T)
    samples = smcmc.sample(N=100, x_sequence=X)
    print(f"Last L samples: \n{samples[:, -1]}")
    # print(np.unique(samples[:, 0], return_counts=True)[1] / len(samples))
    print(f"Mean sample: {samples.mean((0, 1)).round(1)}")


if __name__ == '__main__':
    generate_samples()
