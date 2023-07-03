#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:24:48 2023

@author: vito.dichio

PHASE SPACE EXPLORATION FOR CLUSTER
"""

#%% Import what is needed

import os, time
os.chdir(os.path.dirname(os.path.realpath(__file__)))
start_time = time.time()

import numpy as np
import src.building_blocks as sim

myseed = 160318; np.random.seed(myseed)


#%% parameter ranges

nameset = 'ce_topo_rf'

M = [2**10]
rf =  [20*i for i in range(51,61)]
nseeds = 100

# Copy the above to DA_multi

seed = sim.seeds(nseeds)

grid = [(p1, p2, seed[idx]) for p1 in M for p2 in rf for idx in range(nseeds)]
print('# sims = %i' % len(grid))

#%%
#idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
idx = 0
params = np.array([180,         # N    
                   45,          # T
                   1.4263e-3,   # mu*
                   ##############
                   None,        # M
                   23,          # nu
                   1.0,         # rm
                   None,        # rf
                   None])       # seed

#%%
params[params==None] = grid[idx] ; np.random.seed(grid[idx][-1])

params = np.concatenate((params,sim.shout_prms(which='combo',random=False))) # add fitness parameters

#%%

T, N, M, nu, mu, phi, th, ID = sim.init(params,verbose=True)

outpath, roothpath = sim.shout_paths(ID,simset=nameset)

if nseeds > 1 and idx == 0:
    np.save(roothpath+'/seeds.npy', seed)

cl, ncl, fits = sim.init_pop(M,phi,th)

stats = np.zeros((T+1,sim.statrowlen)); fsts = np.zeros((T+1,2))

#%% Evolution

for i in range(0,T+1):
    
    t = i/nu
    
    ## step 0 - savedata
    stats[i] = sim.save_stats(t, cl, ncl, th, instasavepath=outpath)
    fsts[i] = sim.save_fitstats(fits,ncl)
    
    # step 1 - expand
    pop = np.repeat(cl, ncl, axis=0) 
    
    # step 2 - mutate  
    #mu = sim.mutprofile(i/nu)/nu
    pop = sim.mutations(pop, mu, style = "uphard_mp")  
    
    # step 3 - merge clones
    cl, ncl = np.unique(pop, axis=0, return_counts=True)
    
    # step 4 - selection 
    fits, ncl = sim.selection(t, cl, ncl, M, phi, th) 

    
sim.finalise(outpath, params, stats, fsts)

print("--- %f minutes ---" % ((time.time() - start_time)/60.))

