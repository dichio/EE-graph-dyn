#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 16:33:16 2022

@author: vito.dichio
@version: 3 - UD / ergm terms for fitness function

Last update on 12 Gen 2022
"""

import numpy as np
import collections as cl

myseed = 160318; np.random.seed(myseed)

#%%
#import time

def to_vector(A):
    # ID : -1.a
    # A -> g / tril vectorized (no diag)
    return A[np.tril_indices(A.shape[0], k = -1)]

def to_adj_mat(g):
    # ID : -1.b
    # g -> A
    N = int((1+np.sqrt(1+8*g.size))/2)
    A = np.zeros((N,N))
    A[np.tril_indices(A.shape[0], k = -1)] = g
    return A + A.T


#%% ergm terms
def edges(g):
    return int(np.sum(g))

def twopaths(A,to_adj=False):
    # connected triples 
    if to_adj:
        A = to_adj_mat(A)
    B = np.matmul(A,A)
    return int((np.sum(B)-np.trace(B))/2)

def triangles(A,to_adj=False):
    if to_adj:
        A = to_adj_mat(A)
    return int(np.trace(np.matmul(A,np.matmul(A,A)))/6)

def gwesp(A,decay=0.75,to_adj=False):

    if to_adj:
        A = to_adj_mat(A)
        
    A2_ut = np.triu(np.matmul(A,A),k=1)
    A_esp = A2_ut*A
    
    c_esp = np.array(np.unique(A_esp.flatten(), return_counts=True)) 
    c_esp = np.delete(c_esp,0,1)
    
    gwesp = 0.0
    for k in c_esp.T:
        gwesp+=np.exp(decay)*(1-(1-np.exp(-decay))**k[0])*k[1]
    return gwesp

def gwnsp(A,decay=0.75,to_adj=False):
    if to_adj:
        A = to_adj_mat(A)
        
    A2_ut = np.triu(np.matmul(A,A),k=1)
    A_nsp = A2_ut-A2_ut*A
    
    c_nsp = np.array(np.unique(A_nsp.flatten(), return_counts=True))
    c_nsp = np.delete(c_nsp,0,1)
    
    gwnsp = 0.0
    for k in c_nsp.T:
        gwnsp+=np.exp(decay)*(1-(1-np.exp(-decay))**k[0])*k[1]
    return gwnsp


def gwdegree(A,decay=0.75,to_adj=False): # faster with cl
    if to_adj:
        A = to_adj_mat(A)
        
    degs = np.sum(A,axis=0)
    c_d = cl.Counter(degs)
    del c_d[0]
    gwdegree = 0.0
    for k in c_d.keys():
        gwdegree+=np.exp(decay)*(1-(1-np.exp(-decay))**k)*c_d[k]
    return gwdegree

# load edgecov


dists = to_vector(np.loadtxt('src/data/8_ec_dist.txt',delimiter = " "))


f = 250./1150.
def edgecovdist(g,time = -1):
    if time < 0:
        return np.sum(g*dists)
    else:
        return np.sum(g*dists)*(f+(1-f)/45.*time)


