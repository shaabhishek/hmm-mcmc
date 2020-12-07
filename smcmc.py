import numpy as np
from scipy.special import softmax

from gibbs import compute_gibbs_transition_dist, Gibbs
from hmm import HMM


class SMCMC:
    def __init__(self, model: HMM, T: int):
        self.model = model
        self.T = T

    def jump(self, current_z_sequence, x_sequence):
        (T,) = x_sequence.shape
        assert current_z_sequence.shape == (T - 1,)
        # Pad the old point to make space for $nu_t$
        padded_z_sequence = np.concatenate([current_z_sequence, [0]])
        logits_z = compute_gibbs_transition_dist(T-1, T, padded_z_sequence, x_sequence, self.model)
        padded_z_sequence[T-1] = np.random.choice(np.arange(self.model.num_states), p=softmax(logits_z))
        return padded_z_sequence

    def sample(self, N, x_sequence, initial_sample=None):
        T = self.T
        """
        TODO:
        1. modify gibbs to take a starting point - done
        2. implement jumping function that returns an expanded point after taking the data and previous point - done
        3. 
        4. implement L parallel chains
        """
        samples = [np.array([0], dtype=int)]

        for t in range(2, T+1):
            x_subsequence = x_sequence[:t]
            initial_z_sample = self.jump(samples[-1], x_subsequence)
            for n in range(N):
                gibbs = Gibbs(self.model, t)
                samples = gibbs.sample(10, x_subsequence, initial_sample=initial_z_sample)

        return samples


def generate_samples():
    model = HMM.from_fixed_params()
    T = 6
    Z, X = model.sample(T)
    print(model)
    print(f"X: {X}, True Z: {Z}")

    smcmc = SMCMC(model, T)
    samples = smcmc.sample(N=100, x_sequence=X)
    print(f"Last 10 samples: \n{samples[-10:]}")
    # print(np.unique(samples[:, 0], return_counts=True)[1] / len(samples))
    print(f"Mean sample: {samples.mean(0).round(1)}")


if __name__ == '__main__':
    generate_samples()
