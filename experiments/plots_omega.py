from itertools import product

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

import gibbs
from hmm import HMM
import ipdb
import pickle

def load_mh_prior_transitionmatrix(T,K, model, X):
    # Z, X = loaddata(T, K)
    # with open('experiments/data/hmm_k_{}.pkl'.format(K),'rb') as f:
    #     d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = model.means,scale = model.stds)
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = model.means,scale = model.stds)))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(model.transition_matrix.flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(model.start_prob[z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    print('transition kernel size {}*{}'.format(K ** T, K ** T))
    mh_prior_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_prior_transition_kernel[:,i] = np.minimum(table-table[i],0.)+log_z
        mh_prior_transition_kernel[i,i] = -np.infty
        mh_prior_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_prior_transition_kernel[:,i])))
    return np.exp(mh_prior_transition_kernel)


def load_mh_uniform_transitionmatrix(T,K, model, X):
    #     d = pickle.load(f)
    # p(x|z)
    l = norm.logpdf(np.ones(K)*X[0],loc = model.means,scale = model.stds)
    for i in range(1,T):
        l = list(product(l,norm.logpdf(np.ones(K)*X[i],loc = model.means,scale = model.stds)))
        l = [np.hstack(t) for t in l]
    table = np.sum(np.vstack(l),axis=1)
    # p(z)
    p_z_table = np.log(model.transition_matrix.flatten())
    f = lambda t: t[1]*K+t[0]
    log_z = [np.log(model.start_prob[z[0]]) + np.sum(p_z_table[list(map(f,zip(z[:-1],z[1:])))]) \
         for z in product(np.arange(K),repeat = T)]

    n = K**T
    print('transition kernel size {}*{}'.format(K ** T, K ** T))
    mh_uniform_transition_kernel = np.zeros([n,n])
    for i in range(n):
        mh_uniform_transition_kernel[:,i] = np.minimum(table-table[i]+log_z-log_z[i],0)+np.log(1./n)
        mh_uniform_transition_kernel[i,i] = -np.infty
        mh_uniform_transition_kernel[i,i] = np.log(-np.expm1(logsumexp(mh_uniform_transition_kernel[:,i])))
    return np.exp(mh_uniform_transition_kernel)

def loadmodel(K):
    with open('data/hmm_k_{}.pkl'.format(K), 'rb') as f:
        d = pickle.load(f)

    return HMM(d['num_states'],d['transition_matrix'],d['start_prob'],d['means'],d['stds'])

def compute_eig_decomp(T, K, alg, model, X):
    if alg == 'gibbs':
        rw_matrix = gibbs.transition_kernel(X, model).T  # As defined in the course
    elif alg == 'mh_uniform':
        rw_matrix = load_mh_uniform_transitionmatrix(T, K, model, X)
    elif alg == 'mh_prior':
        rw_matrix = load_mh_prior_transitionmatrix(T, K, model, X)
    w,v = np.linalg.eig(rw_matrix)
    #eigvals = np.linalg.eigvals(rw_matrix)

    #w = eigvals[order]
    #omega = w[1]
    assert np.abs(np.linalg.norm(np.real(w)) - np.linalg.norm(w)) < 1e-10
    w = np.real(w)
    order = np.argsort(w)[::-1]
    w = w[order[1]]


    v = v[:,order[0]]
    assert np.abs(np.linalg.norm(np.real(v)) - np.linalg.norm(v)) < 1e-10
    v = np.real(v)
    v = np.abs(v/v.sum())
    
    #omega = np.real(omega)
    # print(f"T: {T}, K: {K}, Omega: {omega:.5f}")
    return w,v



def collect_results(K,T,N,alg='gibbs'):
    model = loadmodel(K=K)
    kt = dict()
    for i in range(N):
        trial = dict()
        Z,X = model.sample(T=T)
        w1,v1 = compute_eig_decomp(T=T, K=K, alg='gibbs', model=model, X=X)
        w2,v2 = compute_eig_decomp(T=T, K=K, alg='mh_uniform', model=model, X=X)
        w3,v3 = compute_eig_decomp(T=T, K=K, alg='mh_prior', model=model, X=X)
        print (v1[:5],v2[:5],v3[:5])
        assert np.sum(np.abs(v1-v2)) < 1e-10
        assert np.sum(np.abs(v2-v3)) < 1e-10
        trial['pi'] = v1
        trial['w_gibbs'] = w1
        trial['w_mh_uniform'] = w2
        trial['w_mh_prior'] = w3
        kt['n_'+str(i)] = trial
    #ipdb.set_trace()
    return kt




def saveresults(results_df, N):
    results_df.to_csv(f'experiments/data/omega_K_T_df_n_{N}.csv')


def run_experiments(K=2,T=3,N=20):
    results = collect_results(K,T,N)
    #pickle.dump(results,open('results/mix_time/mix_time_analysis_K_{}_T_{}.pkl'.format(K,T),'wb'))
    # saveresults(results, N)
    #results = loadresults(N)
    #plot_results_omega_K(results)
    #plot_results_omega_T(results)
#for t in range(3,5):
import time
start_time = time.time()
run_experiments(K=2,T=13,N=1)


print (time.time()-start_time)
