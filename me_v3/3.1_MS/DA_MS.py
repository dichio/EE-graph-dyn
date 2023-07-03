#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:07:40 2023

@author: vito.dichio
"""


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import src.miner_MS as miner

#%% ONE SIM

simset = 'test'
ID = 'N20-T100-M2048-nu1-mu5.3e-02-rf1.0e+01-d_in1.00-seed160318'

setpath, datapath = miner.shout_paths(ID,simset)

T, ts, N, L, M, nu, mu, phi, E_in = miner.get_params(datapath,verbose=True)

fs, s, cs = miner.import_alltime(datapath); s = s/L

t_ex, s_ex = miner.exact_solution(T, L, nu, mu, phi, E_in)

miner.density_one(t_ex,s_ex,ts,s,"test")

#miner.plot_corrs(ts,cs,L,False)


#%% MULTI SIM

import src.miner_MS as miner

# Comp. time - N

simset = 'test-ct_N'

N = [8*i for i in range(1,15)]

M = [2**(2*i) for i in range(4,8)]

mu = [1e-4]

rf = [5]

nseeds = 10

seeds = miner.sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for idx in range(nseeds)]


comp_time_av, comp_time_sd = miner.build_comp_time(simset,grid)

miner.plot_comp_time(grid, comp_time_av, comp_time_sd, which='N',save=simset)


#%% Comp. time - M

import src.miner_MS as miner

simset = 'test-ct_M'

N = [8,16,32,64]

M = [2**i for i in range(8,16)]

mu = [1e-4]

rf = [5]

nseeds = 10

seeds = miner.sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for idx in range(nseeds)]


comp_time_av, comp_time_sd = miner.build_comp_time(simset,grid)

miner.plot_comp_time(grid, comp_time_av, comp_time_sd, which='M',save=simset)


#%%

# Corr err

simset = 'test-NM'

N = [2**3*i for i in range(1,14)]

M = [256,1024,4096,16384]

mu = [1e-2]

rf = [5e3]

d_trgt = [0.01]

nseeds = 10

seeds = miner.sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, p5, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for p5 in d_trgt for idx in range(nseeds)]


corr_err_av, corr_err_sd = miner.build_corr_err(simset,grid)


#miner.plot_corr_err(grid, corr_err_av, corr_err_sd, which='M',save=simset)
miner.plot_corr_err(grid, corr_err_av, corr_err_sd, which='N',save=simset)
miner.plot_corr_err(grid, corr_err_av, corr_err_sd, which='M/2^L',save=simset)

#miner.plot_corr_err(grid, corr_err_av, corr_err_sd, which='d_trgt',save=simset)


#%%
# solution err

simset = 'test-NM'

N = [2**3*i for i in range(1,14)]

M = [256,1024,4096,16384]

mu = [1e-2]

rf = [5e3]

d_trgt = [0.01]

nseeds = 10

seeds = miner.sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, p5, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for p5 in d_trgt for idx in range(nseeds)]

sol_err_av, sol_err_sd = miner.build_sol_err(simset,grid)

miner.plot_sol_err(grid, sol_err_av, sol_err_sd, which='N',save=simset)


#%%
# solution err

simset = 'test-NM_s_err'

N = [8]

M = [2**(2*i) for i in range(3,10)]

mu = [1e-2]

rf = [5e3]

d_trgt = [0.05*i for i in range(1,20)]

nseeds = 10

seeds = miner.sim.seeds(nseeds)

grid = [(p1, p2, p3, p4, p5, seeds[idx]) for p1 in N for p2 in M for p3 in mu for p4 in rf for p5 in d_trgt for idx in range(nseeds)]

sol_err_av, sol_err_sd = miner.build_sol_err(simset,grid)

miner.plot_sol_err(grid, sol_err_av, sol_err_sd, which='d_trgt',save=simset)


#%%
import src.miner_MRG as miner

simset = 'trials/M'
M = [2**i for i in [5,11,17]]

ref_sim, params, t, s, t_ex, s_ex = miner.get_runs(simset,M,which='M')

miner.plot_runs(ref_sim, simset,params,t,s,t_ex,s_ex,M,which='M',save=simset)


#%%

simset = 'trials/d_trgt'

d_trgt = [0.1,0.2,0.4,0.6]

ref_sim, params, t, s, t_ex, s_ex = miner.get_runs(simset,d_trgt,which='d_trgt')

miner.plot_runs(ref_sim, simset,params,t,s,t_ex,s_ex,d_trgt,which='d_trgt',save=simset)


#%%

simset = 'trials/mu'

mu = [0.001,0.01,0.1]

IDs=["N6-T100-M2048-nu1-mu1.0e-03-rf5.0e+04-d_in0.50-d_trgt0.10-seed160318",
     "N6-T100-M2048-nu1-mu1.0e-02-rf5.0e+03-d_in0.50-d_trgt0.10-seed160318",
     "N6-T100-M2048-nu1-mu1.0e-01-rf5.0e+02-d_in0.50-d_trgt0.10-seed160318"]

ref_sim, params, t, s, t_ex, s_ex = miner.get_runs(simset,mu,which='mu',IDs=IDs)

miner.plot_runs(ref_sim, simset,params,t,s,t_ex,s_ex,mu,which='mu',save=simset)

#%%

simset = 'trials/rho'

rho = [5e2,5e3,5e4]

ref_sim, params, t, s, t_ex, s_ex = miner.get_runs(simset,rho,which='rho')

miner.plot_runs(ref_sim, simset,params,t,s,t_ex,s_ex,rho,which='rho',save=simset)

#%%

miner.plot_ref_sim(simset)






