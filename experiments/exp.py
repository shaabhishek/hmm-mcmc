from gibbs import *
from metropolishastings import *
import numpy as np
from hmm import HMM
import pickle
import argparse
import ipdb

parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('-K', type=int, default = 2)
parser.add_argument('-T', type=int, default = 2)
parser.add_argument('-alg',type=str,default='gibbs')
parser.add_argument('-N',type=int,default=50000)
args = parser.parse_args()
K = args.K
T = args.T
alg = args.alg
N = args.N

with open('data/hmm_k_{}.pkl'.format(K),'rb') as f:
    d = pickle.load(f)
model = HMM(d['num_states'],d['transition_matrix'],d['start_prob'],d['means'],d['stds'])

with open('data/data_t_{}_k_{}.pkl'.format(T,K),'rb') as f:
    Z,X=pickle.load(f)
if alg=='gibbs':
    sampler = Gibbs(model, T,N)
elif alg=='mh_uniform':
    sampler = MH_Uniform(model, T,N)
elif alg=='mh_prior':
    sampler = MH(model, T,N)
else:
    raiseError('Invalid Option')


import pdb; pdb.set_trace()
# s,acc = sampler.sample(X)

# with open('results/{}_k_{}_t_{}.pkl'.format(alg,K,T),'wb') as f:
#     pickle.dump((s,acc),f)
