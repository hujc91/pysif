# pysif
**Py**thon **S**olver for **I**ncompressible **F**luid-dynamics

PySIF is a high performance python implementation of incompressible of Navier–Stokes equations in a triple periodic domain with the pseudo-spectral method. The implementation targets single processor desktop computers. Most codes are accelerated though Numba's Just-In-Time compiler. Fast Fouier transformation employs pyFFTW library, which is a python wrapper for FFTW. The following 512^3 Taylor-Green vortex benchmark simulation is ran on a i7-4790 @ 3.6 GHz (CPU from 2014).
