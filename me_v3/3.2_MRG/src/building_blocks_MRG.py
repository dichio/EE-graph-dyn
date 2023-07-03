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

import time, os, shutil

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

def init(params,verbose=True): ###
    
    # Number of nodes 
    N = int(params[0])
    L = int(N*(N-1)/2)
    
    # Final time
    fT = int(params[1])

    # Number of individuals
    M = int(params[2])
        
    mu = params[4]
    
    if params[3] != None:
        # Time granularity
        nu = params[3]
    else: 
        nu = params[4]
    
    mu_sim = mu/nu
    
    # Simulation steps
    T = int(fT*nu)

    # Selection rate
    phi = mu*params[5]
    phi_sim = phi/nu

    E_in = round(params[6]*L)
    E_trgt = round(params[7]*L)
    
    ID = 'N%i-T%i-M%i-nu%i-mu%.1e-rf%.1e-d_in%.2f-d_trgt%.2f-seed%i' %(params[0],params[1],params[2],nu,params[4],params[5],params[6],params[7],params[8]) 
    
    if verbose:
        print("Parameters")
        print("Exp")
        print("N = %i" %N)
        print("T* = %i" %fT)
        print("mu* = %.2e" %params[4])
        print("")
        
        print("Sim")
        print("M = %i" %M)
        print("nu = %i" %nu)
        print("rho_s = %.1e" %params[5] )
        
        print("--")
        
        print("T = %i" %T)
        print("mu = %.2e (%.2e)" %(mu,mu_sim))
        print("phi = %.2e (%.2e)" %(phi,phi_sim))
        print("d_in (E_in) = %.2f (%i)" %(params[6],E_in))
        print("d_trgt (E_trgt) = %.2f (%i)" %(params[7],E_trgt))
        print("-----")
        
        print("ID :: " + ID)
    
    return T, N, L, M, nu, mu_sim, phi_sim, E_in, E_trgt, ID

def shout_paths(ID,simset=None): ###
    
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


def init_pop(L,M,phi,E_in,E_trgt,how="oneclone"): ###
    
    if how=="oneclone":
        cl = np.zeros((1,L))
        cl[0,np.random.choice(L,size=E_in,replace=False)]=1
        
        ncl = np.array([M],dtype=int)
        
        fit = np.array([fitness(cl,phi,E_trgt)])
    
        return cl, ncl, fit

    elif how=="alldiff":
        cl = np.zeros((M,L))
        for i in range(M):
            cl[i,np.random.choice(L,size=E_in,replace=False)]=1
        
        ncl = np.ones(M,dtype=int)
    
        fit = np.repeat(fitness(cl[0],phi,E_trgt),M)
        
        return cl, ncl, fit
    
def finalise(outpath,params,stats,fits,corrs,allf):

    np.save(outpath+'/params.npy', params )
    np.save(outpath+'/sts.npy', stats )
    np.save(outpath+'/fsts.npy', fits )
    np.save(outpath+'/corrs.npy', corrs )
    np.save(outpath+'/allf.npy', allf )

#%% mutations

def mutations(pop,mu,style="flip"): ###

    if style=="flip":
        (W,L) = pop.shape
        #mat_mut = np.random.poisson(lam=mu, size=(W, L))%2   
        mat_mut = np.random.binomial(1, mu, size=(W, L))
        pop = (pop + mat_mut)%2
        
    if style=="flip_mp":
        (W,L) = pop.shape
        
        for guy in range(W):
            pop[guy,np.random.choice(int(L),size=1)]+=1  
            
        pop = pop%2
        
    if style=="flip":
        (W,L) = pop.shape
        #mat_mut = np.random.poisson(lam=mu, size=(W, L))%2   
        mat_mut = np.random.binomial(1, mu, size=(W, L))
        pop = (pop + mat_mut)%2        
        
    return pop



#%% selection

def fitness(g,phi,E_trgt): ###
    
    L = len(g)
    
    return - phi * 1/L**2 * ( np.sum(g)-E_trgt)**2


def selection(cl, ncl_temp, M, phi, E_trgt): ###
    
    fits = np.apply_along_axis(fitness,1,cl,phi,E_trgt)
    
    pf = np.sum(np.exp(fits)*ncl_temp)
    
    ncl = np.random.multinomial(M, ncl_temp*np.exp(fits)/pf)
    
    return fits, ncl


#%% compute statistics


def save_stats(t, cl, ncl): ###
        
    # Clean
    if t>0:
        cl = cl[ncl!=0,:]
        ncl = ncl[ncl!=0]
    
    # Compute stats
    if len(cl.shape)==1:
        cl = cl.reshape(1,cl.shape[0]) # handle single vector
    s = np.apply_along_axis(np.sum,1,cl) # n. of edges
    
    statrow = np.zeros(2)
    statrow = np.array([ np.average(s,weights=ncl), np.cov(s,fweights=ncl)**(1/2) ]) 
    
    return statrow

def save_fitstats(f,ncl,allf=False): ###
    avf = np.average(f,weights=ncl)
    varf = np.cov(f,bias=True,fweights=ncl)
    if allf == False:
        return avf, varf**(1/2)
    else:
        return np.array([avf, varf**(1/2)]), np.repeat(f, ncl) 
    

import itertools
from random import sample


def corr_indices(L,n_corrs):
    return np.array(sample(list(itertools.combinations(range(L),2)),n_corrs))

def get_corrs(pop, L, n_corrs):
    idx = corr_indices(L,n_corrs)
    x = idx[:,0]; y = idx[:,1]
    n_corrs = len(x)
    corrs = np.zeros(n_corrs)
    
    for i in range(n_corrs):
        s1 = pop[:,x[i]]
        s2 = pop[:,y[i]]
        corrs[i] = np.mean(s1*s2)-np.mean(s1)*np.mean(s2)
    
    return corrs
    
#%% accessories

def tellme(t,every,ncl,fsts,alot=False):
    if t%every == 0:
        if alot == True:
            popsize = np.sum(ncl)            
            print('t=%i, M=%i, avF=%.2e, sdF=%.2e' %(t,         
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
        

def seeds(nseeds=1):
    if nseeds == 1:
        return np.array([160318])
    else:
        np.random.seed(160318)
        return np.random.randint(1e9,size=nseeds)





