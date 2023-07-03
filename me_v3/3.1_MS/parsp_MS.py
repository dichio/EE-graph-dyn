#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27 Mar 2023

@author: vito.dichio
@version: 3.1 - MS 

Last update on 27 Mar 2023
"""

#%% Import what is needed

import os, time
os.chdir(os.path.dirname(os.path.realpath(__file__)))
start_time = time.time()

import numpy as np 

import src.building_blocks_MS as sim

myseed = 160318

#%% init

simset = 'test-NM_ct_M'

N = [8,16,32,64]

M = [2**i for i in range(4,18)]

mu = [1e-1]

rf = [5]

nseeds = 10

seeds = sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for idx in range(nseeds)]

  
######

print('# sims = %i' % len(grid))
#idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
idx = 10

#%%

params = np.array([None,            # N    
                   100,             # T
                   ##############
                   None,            # M
                   1,               # nu
                   None,            # mu
                   None,            # rf
                   0.9,             # d_in
                   None])           # seed

params[params==None] = grid[idx] ; np.random.seed(grid[idx][-1])

#%%

T, N, L, M, nu, mu, phi, E_in, ID = sim.init(params,verbose=True)

outpath = sim.shout_paths(ID,simset=simset)[0]

cl, ncl, fits = sim.init_pop(L,M,phi,E_in,how="alldiff")

n_corrs = 100
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
    pop = sim.mutations(pop, mu, style = "flip")  
    
    # step 3 - merge clones
    cl, ncl = np.unique(pop, axis=0, return_counts=True)
    
    # step 4 - selection 
    fits, ncl = sim.selection(cl, ncl, M, phi)
    #fits = np.zeros(cl.shape[0]) # no selection

    ## step 6 - accessories
    sim.tellme(t,10,ncl,fsts[i],alot=True)
    sim.estimate_simtime(i,T,5)

sim.finalise(outpath, params, stats, fsts, corrs)

print("--- %f minutes ---" % ((time.time() - start_time)/60.))

with open(outpath+'/comptime.txt', 'w') as f:
  f.write('%.1f' % (time.time() - start_time))



