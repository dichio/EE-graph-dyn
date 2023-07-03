#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18 Apr 2023

@author: vito.dichio
@version: 3.1 - MS 

Last update on 18 Apr 2023
"""

#%% Import what is needed

import os, time
os.chdir(os.path.dirname(os.path.realpath(__file__)))
start_time = time.time()

import numpy as np 

import src.building_blocks_MS as sim

myseed = 160318

#%% init

nameset = 'test'

N = 120; L = N*(N-1)/2

params = np.array([N,              # N    
                   100,            # T
                   ##############
                   2**11,          # M
                   1,              # nu
                   1e-2,            # mu
                   10,              # rf
                   1.0,            # d_in
                   160318])     # seed

np.random.seed(int(params[-1]))

T, N, L, M, nu, mu, phi, E_in, ID = sim.init(params,verbose=True)

outpath = sim.shout_paths(ID,simset=nameset)[0]

cl, ncl, fits = sim.init_pop(L,M,phi,E_in,how="alldiff")

n_corrs = 50
stats = np.zeros((T+1,2)); fsts = np.zeros((T+1,2)); corrs = np.zeros((T+1,n_corrs))

#%% evolution 

for i in range(0,T+1):
    
    t = i/nu

    ## step 0 - savedata
    fsts[i] = sim.save_fitstats(fits,ncl)
    stats[i] = sim.save_stats(t, cl, ncl )

    # step 1 - expand
    pop = np.repeat(cl, ncl, axis=0) 
 
    ## step 1.5 - get correlations
    corrs[i] = sim.get_corrs(pop,L,n_corrs)
    
    # step 2 - mutate  
    pop = sim.mutations(pop, 1-np.exp(-mu), style = "flip")  
  
    # step 3 - merge clones
    cl, ncl = np.unique(pop, axis=0, return_counts=True)
 
    # step 4 - selection 
    fits, ncl = sim.selection(cl, ncl, M, phi)
    #fits = np.zeros(cl.shape[0]) # no selection

    ## step 6 - accessories
    sim.tellme(t,10,ncl,fsts[i],alot=True)
    sim.estimate_simtime(i,T,5)

#%% 

sim.finalise(outpath, params, stats, fsts, corrs)

print("--- %f minutes ---" % ((time.time() - start_time)/60.))
      
with open(outpath+'/comptime.txt', 'w') as f:
  f.write('%.1f' % (time.time() - start_time))



