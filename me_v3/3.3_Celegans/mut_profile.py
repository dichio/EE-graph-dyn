#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:18:28 2023

@author: vito.dichio
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import mean_squared_error

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def to_vector(A):
    return A[np.tril_indices(A.shape[0], k = -1)]


N = 180 

L = int(N*(N-1)/2)

texp = np.array([0,5,8,16,23,27,45])
gexp = np.zeros((8,L))

for i in range(8):
    gexp[i,:] = to_vector(np.loadtxt('src/data/'+str(i+1)+'_adj.txt'))

dedges = np.sum(gexp,axis=1)

#%%

def mut_profile(t,E,save=True):
    
    # plot parameters
    mytitlesize = 12
    myfontsize = 7
    mylabelsize = 7

    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(4.*cm,4.*cm))
    
    sns.set_theme()
    
    E_T = E[6:8] 
    E = np.concatenate([E[:6],[np.mean(E[6:8])]])
    
    p, cov = np.polyfit(t, E, 1, cov=True)
    
    d = E/L
    d_fit = (p[1]+p[0]*t)/L
    print("RMSE = %.2E" %mean_squared_error(d, d_fit, squared=False))
    
    c0 = sns.color_palette("tab10").as_hex()[1]
    c1 = sns.color_palette("dark").as_hex()[1]
    c2 = sns.color_palette("dark").as_hex()[0]
    # Plot
    
    plt.plot(t, E[0]/L+(E[6]-E[0])/texp[6]/L * t, label = r'$\sim\mu^*t$',lw = 1, c=c1)
    
    plt.fill_between(t,(p[0]+cov[0][0]**0.5)*t/L+p[1]/L,(p[0]-cov[0][0]**0.5)*t/L+p[1]/L,alpha=.5,lw=.2,edgecolor=c2 )
    
    plt.scatter(t[1:6],E[1:6]/L,c='k',s=15,linewidths=.5, edgecolors='k',zorder=3)
    plt.scatter(t[0],E[0]/L,c=c0,s=20,linewidths=.7, edgecolors='w',zorder=3)
    #plt.scatter(t[6],E[-1]/L,c='w',s=20,linewidths=.7,edgecolor=c0,zorder=3)

    plt.scatter(t[6],E_T[1]/L,c='w',s=20,linewidths=.7,edgecolors=c0,zorder=3)
    plt.scatter(t[6],E_T[0]/L,c='w',s=20,linewidths=.7,edgecolors=c0,zorder=3)




    # Layout
    #plt.xlabel('$t\ [h]$',fontsize=myfontsize)
    plt.ylabel('graph density',fontsize=mylabelsize,labelpad=-6)
    #ax.xaxis.set_label_coords(.95, -0.05)
    
    plt.xticks(fontsize=mylabelsize)
    ax.set_yticks([0.04,0.1])
    plt.yticks(fontsize=mylabelsize)
    
    ax.tick_params(axis='both', which='major', pad=0)
    xticks = [0,5,8,16,23,27, 45]
    xticks_labels = ["0h","","","16h","","27h","45h"]
    plt.xticks(xticks,xticks_labels)
    plt.legend(loc=2,ncol=1,fontsize=mylabelsize)
    
    print("mu = %.2E" %((E[6]-E[0])/t[6]/L) )
    print("mu_fit = %.2E \pm %.2E" %(p[0]/L,1/L*cov[0,0]**0.5 ) )
    
    # Save
    plotpath = './src/data'
    if save == True:
        plt.savefig(plotpath+'/mut_profile.png', dpi=600, bbox_inches = "tight")
    return

mut_profile(texp,dedges)


mus = []
for i in range(6):
    mus.append((dedges[i+1]-dedges[i])/(texp[i+1]-texp[i])/L)
    print("mu_%i = %.5E" %(i,(dedges[i+1]-dedges[i])/(texp[i+1]-texp[i])/L))
    #plt.plot(texp[i:i+2], dedges[i:i+2])






