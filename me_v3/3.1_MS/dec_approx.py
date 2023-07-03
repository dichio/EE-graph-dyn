#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:21:15 2023

@author: vito.dichio
"""

import numpy as np

def to_spin(x):
    return 2*x-1

def to_edge(x):
    return (1+x)/2

N = 10; L = int(N*(N-1)/2)

mu = 1/L
rho = 2e2

#%%
d0 = .9
m0 = to_spin(d0)

lam = 2*L/rho

m1 = lam + (1+lam**2)**(1/2)
m2 = lam - (1+lam**2)**(1/2)

def dmdt(m,L,mu,rho):
    phi = mu*rho
    return -2*mu*m-phi/(2*L)*(1-m**2)
dmdtv = np.vectorize(dmdt)

def mt(t,L,mu,rho):
    return m2*(1+(m1/m2-1)/(1+(m1-m0)/(m0-m2)*np.exp(2*mu*t*(1+lam**(-2))**(1/2))))   
mt = np.vectorize(mt)

#%%

import matplotlib.pyplot as plt
import seaborn as sns

cm = 1/2.54  # centimeters in inches
mylabelsize=9
myfontsize=10

#%% Vector field ode

fig, ax = plt.subplots(figsize=(10*cm,5*cm))

mylabelsize=8
myfontsize=9

xmin = -1.15
xmax = 1.65
epsx = .06
epsy = 1.5e-2

cz = sns.color_palette("dark")
cz1 = sns.color_palette("tab10")

plt.axvspan(xmin, -1., facecolor='0.2', alpha=0.1)
plt.axvspan(1, xmax, facecolor='.2', alpha=0.1)

m = np.arange(-1,1,.01)
plt.plot(m,dmdtv(m,L,mu,rho),zorder=1,label=r'$\dot{m}$',c='k',lw=1)   

m = np.arange(xmin,-1,.01)
plt.plot(m,dmdtv(m,L,mu,rho),zorder=1,c='k',ls='--',lw=.5)  
m = np.arange(1,xmax,.01)
plt.plot(m,dmdtv(m,L,mu,rho),zorder=1,c='k',ls='--',lw=.5)  

plt.scatter(m0,0,s=75,color='w',marker="d",linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_0$', (m0,0), xytext=(m0-epsx,epsy),fontsize=myfontsize)
plt.scatter(m1,0,s=75,color=cz[3],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_1$', (m1,0), xytext=(m1-2*epsx, epsy),fontsize=myfontsize)
plt.scatter(m2,0,s=75,c=cz[2],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$m_2$', (m2,0), xytext=(m2, epsy),fontsize=myfontsize)

#X = np.arange(-1,1,0.2)
#Y = np.repeat(0,len(X))
#plt.quiver([X,Y],[np.sign(dmdtv(X,L,mu,rho),Y)])

x_pos = np.arange(-1,1.1,0.25)
y_pos = [0 for i in range(len(x_pos))]

x_direct = dmdtv(x_pos,L,mu,rho)

ax.quiver(x_pos, y_pos, x_direct, y_pos, alpha=1, scale=1, width=5e-3,headwidth=4,headlength=4,color=cz[2],zorder=3)       
       
plt.legend(loc=9)
#plt.xlabel('$m$',fontsize=myfontsize)
#ax.xaxis.set_label_coords(1., .55)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

ax.set_xticks([-1.0,1.0])

plt.xticks(fontsize=mylabelsize)
plt.yticks(fontsize=mylabelsize)
ax.set(yticklabels=[])  # remove the tick labels
ax.tick_params(left=False)  # remove the ticks

plt.savefig('output/dec_approx/dmdt.png', dpi=1200, bbox_inches = "tight") 


#%% solution

fig, ax = plt.subplots(figsize=(15*cm,7.5*cm))

t = np.arange(1,100,1)

epsx = 1

d0 = .9
m0 = to_spin(d0)

cz = sns.color_palette("tab10")


plt.plot(t,to_edge(mt(t,L,mu,rho)),c=cz[0],label=r'$\bar{\mu}$')
plt.plot(t,to_edge(mt(t,L,2*mu,rho)),c=cz[1],label=r'$2\bar{\mu}$')
plt.plot(t,to_edge(mt(t,L,3*mu,rho)),c=cz[5],label=r'$3\bar{\mu}$')


cz = sns.color_palette("dark")
plt.scatter(0,to_edge(m0),s=150,color='w',marker="d",linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$d_0$', (0,to_edge(m0)), xytext=(2*epsx, to_edge(m0)+epsy))
plt.scatter(0,to_edge(m2),s=150,c=cz[2],linewidth=1,edgecolors='k',zorder=3)
ax.annotate('$d_2$', (0,to_edge(m2)), xytext=(2*epsx, to_edge(m2)+epsy))

plt.legend(loc=4,ncol=3)
plt.ylim([0,1])

ax.set_yticks([.5,1.0])

plt.xlabel('t',fontsize=myfontsize)
ax.xaxis.set_label_coords(1., -0.1)
plt.ylabel('$d_t$',fontsize=myfontsize,rotation=0)
ax.yaxis.set_label_coords(0.05, 1.05)  

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 0))
#ax.spines['bottom'].set_position(('data', 0))

plt.xticks(fontsize=mylabelsize)
plt.yticks(fontsize=mylabelsize)

plt.savefig('output/dec_approx/mt.png', dpi=1200, bbox_inches = "tight") 
