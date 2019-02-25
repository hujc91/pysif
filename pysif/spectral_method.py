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

    def initialization(self, ic_file):
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

        with h5py.File(self.path + ic_file, 'r') as f:
            self.R1p = f['V'][:]
        
        self.vector_rfft(self.R1s, self.R1p)

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
    def solve_current_time_step(self, N_current):
        self.N_current = N_current
        
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
# Post-processing Functions
# --------------------------------------------------------------------------------------
    def zero_vector_field_physical(self):
        '''Create a zero vector field in physical domain for post-processing

        Output:
        Vp - zero vector field in physical domain
        '''
        physical_array_size = (3, self.N[0], self.N[1], self.N[2])
        Vp = np.zeros(physical_array_size)
        return Vp

    def zero_vector_field_spectral(self):
        '''Create a zero vector field in spectral domain for post-processing

        Output:
        Vs - zero vector field in spectral domain
        '''
        spectral_array_size = (3, self.N[0], self.N[1], int(self.N[2]/2+1))
        Vs = np.zeros(spectral_array_size, dtype='complex128')
        return Vs

    def create_mesh_grid(self):
        N = self.N
        L = self.L

        x = np.arange(N[0])*L[0]/N[0]-L[0]/2
        y = np.arange(N[1])*L[1]/N[1]-L[1]/2
        z = np.arange(N[2])*L[2]/N[2]-L[2]/2
        
        X = np.array(np.meshgrid(x,y,z,indexing='ij'))
        
        return X
        
    def user_apply_dealiasing_self(self, Rs):
        jit_function.apply_dealiasing_self(Rs, self.d0, self.d1, self.d2)
    
    def solve_poisson(self, Rs):
        jit_function.solve_poisson(Rs, self.k0, self.k1, self.k2)
        
    def curl(self, Rs):
        jit_function.curl(Rs, self.k0, self.k1, self.k2)

    def current_time(self):
        return self.N_current*self.dt

    def compute_vorticity_field(self, register):
        if register == 'R1':
            jit_function.curl_post(self.R3s, self.R1s, self.k0, self.k1, self.k2)
            self.vector_irfft(self.R1p, self.R3s)

        elif register == 'R2':
            jit_function.curl_post(self.R3s, self.R1s, self.k0, self.k1, self.k2)
            self.vector_irfft(self.R2p, self.R3s)
        
        else:
            raise Exception('Incorrect register, must be R1 or R2')

    def compute_velocity_field(self, register):
        if register == 'R1':
            self.vector_irfft(self.R1p, self.R1s)

        elif register == 'R2':
            self.vector_irfft(self.R2p, self.R1s)
        
        else:
            raise Exception('Incorrect register, must be R1 or R2')

    def conpute_total_enstrophy(self, register):
        if register == 'R1':
            ens = jit_function.get_total_enstrophy(self.R1p, self.dV)

        elif register == 'R2':
            ens = jit_function.get_total_enstrophy(self.R2p, self.dV)

        else:
            raise Exception('Incorrect register, must be R1 or R2')
        
        return ens

    def compute_magnitude(self, output, register):
        if register == 'R1':
            jit_function.get_magnitude(output, self.R1p)

        elif register == 'R2':
            jit_function.get_magnitude(output, self.R2p)

        else:
            raise Exception('Incorrect register, must be R1 or R2')

# --------------------------------------------------------------------------------------
# I/O Function
# --------------------------------------------------------------------------------------
    def save_to_xdmf(self, data_dict):
            
            n  = len(str(self.N_itr))
            dL = self.L/self.N
            fname = f'{self.path}sol-itr-{str(self.N_current).zfill(n)}'

            with open(f'{fname}.xmf','w') as fx, h5py.File(f'{fname}.h5', 'w') as fh:
                fx.write('<?xml version="1.0" ?>\n\n'
                         '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n\n'
                         '<Xdmf Version="2.0">\n\n'
                         '  <Domain>\n\n'
                         '    <Grid Name="mesh" GridType="Uniform">\n\n'
                        f'      <Time Value="{self.current_time()}"/>'
                        f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{self.N[2]} {self.N[1]} {self.N[0]}"/>\n\n'
                         '      <Geometry GeometryType="ORIGIN_DXDYDZ">\n\n'
                         '        <DataItem DataType="Float" Dimensions="3" Format="XML">\n'
                        f'          {-self.L[2]/2} {-self.L[1]/2} {-self.L[0]/2}\n'
                         '        </DataItem>\n\n'
                         '        <DataItem DataType="Float" Dimensions="3" Format="XML">\n'
                        f'          {dL[2]} {dL[1]} {dL[0]}\n'
                         '        </DataItem>>\n\n'
                         '      </Geometry>\n\n')
                
                for key,item in data_dict.items():
                    fx.write(f'      <Attribute Name="{key}" AttributeType="Scalar" Center="Node">\n\n'
                             f'        <DataItem Dimensions="{self.N[2]} {self.N[1]} {self.N[0]}" NumberType="Float" Precision="4" Format="HDF">\n'
                             f'          sol-itr-{str(self.N_current).zfill(n)}.h5:/{key}\n'
                              '        </DataItem>\n\n'
                              '      </Attribute>\n\n')
                    
                    fh.create_dataset(key, data=item.flatten('F'), compression=0)

                fx.write('    </Grid>\n\n'
                         '  </Domain>\n\n'
                         '</Xdmf>\n\n')
