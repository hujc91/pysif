
#%%
import h5py
import numpy as np


#%%
N = np.ones(3, dtype='int')*2**9
L = np.ones(3)*2*np.pi


#%%
# Physical Grid
x = np.arange(N[0])*L[0]/N[0]
y = np.arange(N[1])*L[1]/N[1]
z = np.arange(N[2])*L[2]/N[2]
X = np.array(np.meshgrid(x,y,z,indexing='ij'))


#%%
# Taylor-Green Initial Condition
U_ic    =  np.zeros((3, N[0], N[1], N[2]))
U_ic[0] =  np.sin(X[0])*np.cos(X[1])*np.cos(X[2])
U_ic[1] = -np.cos(X[0])*np.sin(X[1])*np.cos(X[2])
U_ic[2] =  0.0


#%%
with h5py.File('U_ic.hdf5', 'w') as f:
    f['V'] = U_ic
