'''Numba Just-In-Time compiled functions for optimal performance
'''
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def apply_dealiasing(Us_out, Us_in, d0, d1, d2):
    '''Apply dealiasing to velocity field
    '''
    for i in range(len(d0)):
        for j in range(len(d1)):
            for k in range(len(d2)):
                d = d0[i]*d1[j]*d2[k]
                Us_out[0,i,j,k] = d*Us_in[0,i,j,k]
                Us_out[1,i,j,k] = d*Us_in[1,i,j,k]
                Us_out[2,i,j,k] = d*Us_in[2,i,j,k]

@jit(nopython=True, parallel=True)
def curl(Rs, k0, k1, k2):
    '''Obtain the vorticity field from the velocity field in spectral domain
    '''
    # Loop though every array entry
    for i in range(len(k0)):
        for j in range(len(k1)):
            for k in range(len(k2)):

                # Load array entry
                K0 = k0[i]
                K1 = k1[j]
                K2 = k2[k]
                U0 = Rs[0,i,j,k]
                U1 = Rs[1,i,j,k]
                U2 = Rs[2,i,j,k]

                # Perform curl operation
                Rs[0,i,j,k] = 1j*(K1*U2-K2*U1)
                Rs[1,i,j,k] = 1j*(K2*U0-K0*U2)
                Rs[2,i,j,k] = 1j*(K0*U1-K1*U0)

@jit(nopython=True, parallel=True)
def cross(R1p, R2p):
    '''Obtain nonlinear convection term in physical domain
    '''
    # Loop though every array entry
    for i in range(R1p.shape[1]):
        for j in range(R1p.shape[2]):
            for k in range(R1p.shape[3]):

                # Load array entry
                U0 = R1p[0,i,j,k]
                U1 = R1p[1,i,j,k]
                U2 = R1p[2,i,j,k]
                W0 = R2p[0,i,j,k]
                W1 = R2p[1,i,j,k]
                W2 = R2p[2,i,j,k]

                # Perform cross product
                R1p[0,i,j,k] = U1*W2-U2*W1
                R1p[1,i,j,k] = U2*W0-U0*W2
                R1p[2,i,j,k] = U0*W1-U1*W0

@jit(nopython=True, parallel=True)
def get_RHS(R2s, Cs, k0, k1, k2, nu):
    '''Obtain right hand side of the Navier-Stokes momentum equation
    '''
    K2_temp = 0.

    # Loop though every array entry
    for i in range(len(k0)):
        for j in range(len(k1)):
            for k in range(len(k2)):
                
                # Compute spectral operators
                K_0 = k0[i]
                K_1 = k1[j]
                K_2 = k2[k]
                                
                K2 = K_0**2 + K_1**2 + K_2**2

                if K2 == 0: 
                    K2_temp = 1
                else:
                    K2_temp = K2

                KK2_0 = K_0/K2_temp
                KK2_1 = K_1/K2_temp
                KK2_2 = K_2/K2_temp

                # Load convection term since it will be used twice
                C0 = Cs[0,i,j,k]
                C1 = Cs[1,i,j,k]
                C2 = Cs[2,i,j,k]

                # Pressure term
                Ps = C0*KK2_0 + C1*KK2_1 + C2*KK2_2

                # Change in velocity vector field
                R2s[0,i,j,k] = C0 - nu*K2*R2s[0,i,j,k] - Ps*K_0
                R2s[1,i,j,k] = C1 - nu*K2*R2s[1,i,j,k] - Ps*K_1
                R2s[2,i,j,k] = C2 - nu*K2*R2s[2,i,j,k] - Ps*K_2

@jit(nopython=True, parallel=True)
def rk_update_R1s(R1s, R2s, coeff):
    '''Update register 1 for RK method
    '''
    # Loop though every array entry
    for i in range(R1s.shape[1]):
        for j in range(R1s.shape[2]):
            for k in range(R1s.shape[3]):
                R1s[0,i,j,k] = R1s[0,i,j,k] + coeff*R2s[0,i,j,k]
                R1s[1,i,j,k] = R1s[1,i,j,k] + coeff*R2s[1,i,j,k]
                R1s[2,i,j,k] = R1s[2,i,j,k] + coeff*R2s[2,i,j,k]

@jit(nopython=True, parallel=True)
def rk_update_R2s(R2s, R1s, coeff):
    '''Update register 2 for RK method
    '''
    # Loop though every array entry
    for i in range(R1s.shape[1]):
        for j in range(R1s.shape[2]):
            for k in range(R1s.shape[3]):
                R2s[0,i,j,k] = R1s[0,i,j,k] + coeff*R2s[0,i,j,k]
                R2s[1,i,j,k] = R1s[1,i,j,k] + coeff*R2s[1,i,j,k]
                R2s[2,i,j,k] = R1s[2,i,j,k] + coeff*R2s[2,i,j,k]

@jit(nopython=True, parallel=True)
def curl_post(R3s, R1s, k0, k1, k2):
    '''Obtain the vorticity field from the velocity field in spectral domain
    '''
    # Loop though every array entry
    for i in range(len(k0)):
        for j in range(len(k1)):
            for k in range(len(k2)):

                # Load array entry
                K0 = k0[i]
                K1 = k1[j]
                K2 = k2[k]
                U0 = R1s[0,i,j,k]
                U1 = R1s[1,i,j,k]
                U2 = R1s[2,i,j,k]

                # Perform curl operation
                R3s[0,i,j,k] = 1j*(K1*U2-K2*U1)
                R3s[1,i,j,k] = 1j*(K2*U0-K0*U2)
                R3s[2,i,j,k] = 1j*(K0*U1-K1*U0)

@jit(nopython=True, parallel=True)
def get_total_enstrophy(Wp, dV):
    '''Obtain total enstrophy of the flow field

    Input:
    Wp - vortictiy vector field in physical domain

    Output
    ens - total enstrophy
    '''
    ens = 0.0
    
    for l in range(Wp.shape[0]):
        for i in range(Wp.shape[1]):
            for j in range(Wp.shape[2]):
                for k in range(Wp.shape[3]):
                    ens = ens + Wp[l,i,j,k]**2

    ens = 0.5*dV*ens
    
    return ens
