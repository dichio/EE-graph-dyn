#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:18:54 2022

@author: vito.dichio
@version: 3 - UD / Master-Equation building blocks

Last update on 12 Gen 2022
"""

#ehj 

import numpy as np

import time, os, shutil, sys

import src.ergm_terms as ergm

myseed = 160318; np.random.seed(myseed)

#%% graph vec - adj

def to_vector(A,k=-1):
    return A[np.tril_indices(A.shape[0], k)]

def to_adj_mat(g,N=180,k=-1):
    A = np.zeros((N,N))
    A[np.tril_indices(A.shape[0], k = k)] = g
    if k == -1:
        return A + A.T
    elif k ==0:
        return A + A.T - np.diag(np.diag(A))

#%% in / out

def init(params,verbose=True):
    
    # Number of nodes 
    N = int(params[0])
    
    # Final time
    fT = int(params[1])
    
    # mutation rate (upper bound)
    mu_ub = params[2]

    # Number of individuals
    M = int(params[3])

    # Time granularity
    nu = int(params[4])

    # Simulation steps
    T = int(fT*nu)

    # Mutation rates 
    mu = mu_ub * params[5]/nu

    # Selection rate
    phi = mu_ub * params[6]/nu
    
    # Fitness parameters
    th = params[8:]

#    ID = 'M%i-nu%i-rm%.4f-rf%.2f-seed%i' %(params[3],params[4],params[5],params[6],params[7]) 
    ID = 'M%i-rf%.2f-seed%i' %(params[3],params[6],params[7]) 

    if verbose:
        print("Parameters")
        print("Exp")
        print("N = %i" %N)
        print("T* = %i" %fT)
        print("mu* = %.2e" %(mu_ub))
        print("")
        print("fit func = %.3f * gwdegree(tau = %.3f) + %.3f * gwesp(tau = %.3f) " %(th[0],th[1],th[2],th[3]))
        print("")
        
        print("Sim")
        print("M = %i" %M)
        print("nu = %i" %nu)
        print("rho_m = %.2f" %params[5] )
        print("rho_s = %.1f" %params[6] )
        
        print("--")
        
        print("T = %i" %T)
        print("mu = %.2e" %mu)
        print("phi = %.2e" %phi)
        
        print("-----")
        
        print("ID :: " + ID)
    
    return T, N, M, nu, mu, phi, th, ID

def shout_paths(ID,simset=None):
    
    if simset != None:
        rootpath = './output/' + simset + '/'
    else: 
        rootpath = './output/'
    
    if not(os.path.exists(rootpath)):
        os.makedirs(rootpath)
        
    outpath = rootpath + ID
    
    if os.path.exists(outpath):
         shutil.rmtree(outpath)
    os.mkdir(outpath)
    
    return outpath, rootpath


def init_pop(M,phi,th):

    g_init = to_vector(np.loadtxt('src/data/1_adj.txt')) # Load first network
    cl = np.array([g_init])
    ncl = np.array([M],dtype=int)
    fit = np.array([fitness(g_init,phi,th)])
    return cl, ncl, fit

def finalise(outpath,params,stats,fits):

    np.save(outpath+'/params.npy', params )
    np.save(outpath+'/sts.npy', stats )
    np.save(outpath+'/fsts.npy', fits )

#%% mutations

def mutations(pop,mu,style="flip"):
    
    if style=="flip":
        (W,L) = pop.shape
        mat_mut = np.random.poisson(lam=mu, size=(W, L))%2     
        pop = (pop + mat_mut)%2
        
    if style=="upsoft": # inaccurate for denser graphs but faster
        (W,L) = pop.shape
        for guy in range(W):
            nmut = np.random.poisson(lam=mu*L)
            if nmut !=0:
                pop[guy,np.random.choice(np.arange(L), size=nmut, replace=False)] = 1
   
    if style=="uphard":
        (W,L) = pop.shape
        for guy in range(W):
            nmut = np.random.poisson(lam=mu*L)
            if nmut !=0:
                growth = pop[guy,pop[guy,:]==0]
                growth[np.random.choice(np.arange(np.size(growth)), size=nmut, replace=False)] = 1
                pop[guy,pop[guy]==0] = growth

    if style=="uphard_mp":
        (W,L) = pop.shape
        
        if mu*L > 1:
            sys.exit("uphard_mp not possible: mu*L > 1, increase nu!")
            
        for guy in range(W):
            mut = np.random.binomial(1,mu*L)
            if mut == 1:
                growth = pop[guy,pop[guy,:]==0]
                growth[np.random.choice(np.arange(np.size(growth)), size=1, replace=False)] = 1
                pop[guy,pop[guy]==0] = growth
                
    if style=="downhard":
        (W,L) = pop.shape
        for guy in range(W):
            nmut = np.random.poisson(lam=mu*L)
            if nmut !=0:
                loss = pop[guy,pop[guy,:]==1]
                loss[np.random.choice(np.arange(np.size(loss)), size=nmut, replace=False)] = 0
                pop[guy,pop[guy]==1] = loss
    return pop

#%% selection
prms_avs = np.array([[0.45253, 1.90960, 0.62583, 1.43228], # ID = 7
                    [0.43384, 1.97090, 0.52931, 1.54230]]) # ID = 8

prms_sds = np.array([[0.19578, 0.45828, 0.05618 , 0.06651],
                    [0.19724, 0.48299, 0.04815, 0.07488]])

def shout_prms(which='combo', random=False):
    if which == 'combo':
        avs = np.mean(prms_avs,axis=0)
        sds = (np.sum(prms_sds**2,axis=0)/4)**(1/2)
    else:
        avs = prms_avs[which-7]
        sds = prms_sds[which-7]

    if random:
        return np.random.normal(avs, sds)
    else:
        return avs
    
    
    
def fitness(g,phi,th,t=-1):
    A = to_adj_mat(g) 
    return phi * ( 
                    th[0] * ergm.gwdegree(A,decay=th[1],to_adj=False) +
                    th[2] * ergm.gwesp(A,decay=th[3],to_adj=False)
                    )

def selection_noisy(t, cl, ncl_temp, phi, th, cc = None):
    W = cl.shape[0]
    
    fits = np.apply_along_axis(fitness,1,cl,phi,th,t)
    
    avf = np.average(np.exp(fits), weights = ncl_temp)
    
    if cc == None:
        red = 0.0
    else:
        red = 1.0 - np.sum(ncl_temp)/cc # -- keeps the popsize close to carrying capacity (cc)
    
    ncl = []
    for i in range(W): 
        ncl.append(
            np.random.poisson( ncl_temp[i] * np.exp(fits[i] + red) / avf )
            )
    ncl = np.array(ncl)
    
    return fits, ncl

def selection(t, cl, ncl_temp, M, phi, th):
    
    fits = np.apply_along_axis(fitness,1,cl,phi,th,t)
    
    pf = np.sum(np.exp(fits)*ncl_temp)
    
    ncl = np.random.multinomial(M, ncl_temp*np.exp(fits)/pf)
    
    return fits, ncl

#%% compute statistics

nstats = 3

def compute_stats(cl,th,t=-1):

    if len(cl.shape)==1:
        cl = cl.reshape(1,cl.shape[0]) # handle single vector

    W = cl.shape[0]
    
    statvalues = np.zeros((W,nstats))
    for i in range(W):
        cl_v = cl[i]
        cl_adj = to_adj_mat(cl[i])
        statvalues[i] = ergm.edges(cl_v),                                \
                        ergm.gwdegree(cl_adj,decay=th[1],to_adj=False),  \
                        ergm.gwesp(cl_adj,decay=th[3],to_adj=False)
    return statvalues


# Experimental time stamps
T_save = [5,8,16,23,27,45]

statrowlen = int((nstats**2+3*nstats)/2)

def save_stats(t, cl, ncl, th, instasavepath = None, save_cl = False):
        
    # Clean
    if t>0:
        cl = cl[ncl!=0,:]
        ncl = ncl[ncl!=0]
    
    # Compute stats
    s = compute_stats(cl,th,t)
    
    if instasavepath != None and t in T_save:
        np.save(instasavepath+'/T%i_sts.npy' %t, s )
        np.save(instasavepath+'/T%i_nc.npy' %t, ncl )
        if save_cl:
            np.save(instasavepath+'/T%i_cl.npy' %t, cl )
    
    statrow = np.zeros(statrowlen)
    statrow[:nstats] = np.average(s,axis=0,weights=ncl) # store means
    if ncl.size > 1:
        # store covariance matrix (low tri)
        statrow[nstats:] = to_vector(np.cov(s,rowvar=False,bias=True,fweights=ncl),k=0)
    
    return statrow

def save_fitstats(f,ncl):
    avf = np.average(f,weights=ncl)
    varf = np.cov(f,bias=True,fweights=ncl)
    return avf,varf



#%% accessories

def check_popsize(ncl,M):
    popsize = np.sum(ncl)
    if popsize < 0.05*M:
        print('Less than the 5% of individuals left')
    if popsize > 10*M:
        print('>10x increase of population size')

def tellme(t,every,ncl,fsts,alot=False):
    if t%every == 0:
        if alot == True:
            popsize = np.sum(ncl)            
            print('t=%i, M=%i, avF=%.2f, sdF=%.4f' %(t,         
                                                     popsize,   
                                                     fsts[0],  
                                                     fsts[1]))
        else:
            print('t = %i' %t)

def estimate_simtime(i,T=45,twin=1):
    global allez
    if i == 2:
        allez = time.time()
    if i == 2+twin:
        print("")
        print("It'll take approx %.1f min more" % ((T-twin)*(time.time() - allez)/(60.*twin)))
        print("")
        
def mut_rate_adjust(t,nu,mu0,pop):
    L = pop.shape[1]
    target = 1651.0
    de = target - np.mean(np.sum(pop,axis=1))
    if de < 0:
        return 0
    mu = de/(45+1-t)/L/nu
    return mu

def seeds(nseeds=1):
    if nseeds == 1:
        return np.array([160318])
    else:
        return np.random.randint(1e9,size=nseeds)





