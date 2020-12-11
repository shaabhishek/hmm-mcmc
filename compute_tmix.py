import numpy as np

from experiment_analysis import loadpi, loaddatamodel
from plots_omega import compute_omega

tmix = lambda eps,pi,omega: (np.log(1/eps) + .5 * np.log(1/np.min(pi))) / (1-omega)

def do_experiments():
    alg = 'gibbs'
    eps = 0.01
    for K in range(2, 7):
        for T in range(2, 14):
            if K**T > 2000: continue #REMOVE THIS DURING FINAL RUN
            try:
                pi = loadpi(T=T, K=K, alg='gibbs')
                (_, X), model = loaddatamodel(T, K)
                omega = compute_omega(T, K, alg, model, X)
                print(f"T_mix for K:{K}, T:{T}: {tmix(eps, pi, omega)}")
            except:
                continue


if __name__ == '__main__':
    do_experiments()