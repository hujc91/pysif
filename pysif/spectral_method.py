import h5py
import pyfftw
import numpy as np
import configparser
from tqdm import tqdm
from pysif import jit_function

class solver:
    '''Fluipy solver for triple periodic incompressible Navier-Stokes equations
    
    Spatial discretization  - pseudo spectral method
    Temporal discretization - Third order low storage Runge Kutta method (RK3-2R)
    '''

    def __init__(self, config_file_path):
        # Load configuration file
        config = configparser.ConfigParser()
        config.read(config_file_path)

        # Set fluid property
        self.nu = float(config['FLUID_PROPERTY']['viscosity'])

        # Set solver control
        sc = config['SOLVER_CONTROL']
        N  = np.array((int(sc['nx']), int(sc['ny']), int(sc['nz'])), dtype='int')
        L  = np.array((float(sc['lx']), float(sc['ly']), float(sc['lz'])), dtype='float')

        self.N     = N
        self.L     = L
        self.dV  = np.prod(self.L/self.N)
        self.dt    = float(config['SOLVER_CONTROL']['dt'])
        self.N_itr = int(config['SOLVER_CONTROL']['iterations'])

        self.fftw_planning(int(config['SOLVER_CONTROL']['cores']))

        # Set write control
        self.N_write = int(config['WRITE_CONTROL']['iterations'])
        self.path    = config['WRITE_CONTROL']['directory']

        # Generate spectral domain wavenumnbers and dealiasing operator
        scale = 2*np.pi/L
        kmax  = (N/2+1)*scale

        self.k0 = np.fft.fftfreq(N[0],1/N[0])*scale[0]
        self.k1 = np.fft.fftfreq(N[1],1/N[1])*scale[1]
        self.k2 = np.fft.fftfreq(N[2],1/N[2])[:int(N[2]/2)+1]*scale[2]

        self.d0 = np.exp(-36*(1.2*np.abs(self.k0)/kmax[0])**36)
        self.d1 = np.exp(-36*(1.2*np.abs(self.k1)/kmax[1])**36)
        self.d2 = np.exp(-36*(1.2*np.abs(self.k2)/kmax[2])**36)

    def initialization(self):
        '''Initialize all nessessary arrays for computation
        '''
        # Array sizes
        physical_array_size = (3, self.N[0], self.N[1], self.N[2])
        spectral_array_size = (3, self.N[0], self.N[1], int(self.N[2]/2+1))

        # Initialize physical arrays
        self.R1p = np.zeros(physical_array_size)
        self.R2p = np.zeros(physical_array_size)

        # Initialize spectral arrays
        self.R1s = np.zeros(spectral_array_size, dtype='complex128')
        self.R2s = np.zeros(spectral_array_size, dtype='complex128')
        self.R3s = np.zeros(spectral_array_size, dtype='complex128')

        # Load initial condition
        self.N_current = 0 

        with h5py.File(self.path+'U_ic.hdf5', 'r') as f:
            self.R1p = f['V'][:]
        
        self.vector_rfft(self.R1s, self.R1p)
        self.post_processing()

# --------------------------------------------------------------------------------------
# FFT Functions
# --------------------------------------------------------------------------------------
    def fftw_planning(self, N_cores):
        '''FFTW planning for fastest transformation
        '''
        N = self.N

        fftw_p = pyfftw.empty_aligned((N[0], N[1], N[2]), dtype='float64')
        fftw_s = pyfftw.empty_aligned((N[0], N[1], int(N[2]/2+1)),
                                           dtype='complex128')
                                           
        self.rfft  = pyfftw.FFTW(fftw_p, fftw_s, axes=(0,1,2), threads=N_cores,
                                 direction='FFTW_FORWARD',
                                 flags=('FFTW_DESTROY_INPUT','FFTW_MEASURE'))
        self.irfft = pyfftw.FFTW(fftw_s, fftw_p, axes=(0,1,2), threads=N_cores,
                                 direction='FFTW_BACKWARD',
                                 flags=('FFTW_DESTROY_INPUT','FFTW_MEASURE'))

    def vector_rfft(self, Vs, Vp):
        '''Perform real valued FFT of a vector field with pyFFTW
        '''
        for i in range(3):
            self.rfft.input_array[:] = Vp[i]
            Vs[i] = self.rfft()

    def vector_irfft(self, Vp, Vs):
        '''Perform real valued inverse FFT of a vector field with pyFFTW
        '''
        for i in range(3):
            self.irfft.input_array[:] = Vs[i]
            Vp[i] = self.irfft()

# --------------------------------------------------------------------------------------
# Compute dU_dt = f(U)
# --------------------------------------------------------------------------------------
    def dU_dt(self):
        # Step 1: velocity field for convection term calculation
        jit_function.apply_dealiasing(self.R3s, self.R2s, self.d0, self.d1, self.d2)
        self.vector_irfft(self.R1p, self.R3s)

        # Step 2: vorticity field for convection term calculation
        jit_function.curl(self.R3s, self.k0, self.k1, self.k2)
        self.vector_irfft(self.R2p, self.R3s)

        # Step 3: compute convection term
        jit_function.cross(self.R1p, self.R2p)
        self.vector_rfft(self.R3s, self.R1p)

        # Step 4: compute dU_dt
        jit_function.get_RHS(self.R2s, self.R3s, self.k0, self.k1, self.k2, self.nu)

# --------------------------------------------------------------------------------------
# RK Time Stepping
# --------------------------------------------------------------------------------------
    def RK_time_stepping(self):
        
        self.R2s[:] = self.R1s
        
        # Stage 1:
        self.dU_dt()
        jit_function.rk_update_R1s(self.R1s, self.R2s, 1/4*self.dt)

        # Stage 2:
        jit_function.rk_update_R2s(self.R2s, self.R1s, (8/15-1/4)*self.dt)
        self.dU_dt()

        # stage 3:
        jit_function.rk_update_R2s(self.R2s, self.R1s, 5/12*self.dt)
        self.dU_dt()
        jit_function.rk_update_R1s(self.R1s, self.R2s, 3/4*self.dt)

# --------------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------------
    def solve(self):
        for N_current in tqdm(range(1,self.N_itr+1)):
            self.N_current = N_current
            self.RK_time_stepping()
            self.post_processing()

# --------------------------------------------------------------------------------------
# Post-processing Functions
# --------------------------------------------------------------------------------------
    def post_processing(self):
        if self.N_current % self.N_write == 0:

            # Enstrophy
            if self.N_current == 0:
                f = open(self.path + 'enstrophy.csv','w')
                f.close()

            jit_function.curl_post(self.R3s, self.R1s, self.k0, self.k1, self.k2)
            self.vector_irfft(self.R1p, self.R3s)
            total_enstrophy = jit_function.get_total_enstrophy(self.R1p, self.dV)

            with open(self.path + 'enstrophy.csv','a+') as f:
                f.write(f'{self.N_current*self.dt},{total_enstrophy}\n')
            
            Wm = np.sqrt(np.sum(self.R1p[:,0,:,:]**2, axis=0))
            
            save_path = self.path+'W-x0-itr-'+str(self.N_current).zfill(4)+'.hdf5'
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('V', data=Wm, compression=0)

