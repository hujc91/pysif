# PySIF
**Py**thon **S**olver for **I**ncompressible **F**luid-dynamics

Author: Jia Cheng Hu (jia.cheng.hu@uwaterloo.ca)

PySIF* is a high performance python implementation of incompressible of Navier–Stokes equations in a triple periodic domain with the pseudo-spectral method. The implementation targets single processor desktop computers. Most codes are accelerated though Numba's Just-In-Time compiler. Fast Fouier transformation employs pyFFTW library, which is a python wrapper for FFTW. The following 512^3 Taylor-Green vortex benchmark simulation ran on an i7-4790 @ 3.6 GHz (a CPU from 2014) with less than 20 GB of RAM, and it only takes 19 wall clock hours to finish. The reference data is obtained from the 2nd International Workshop on High‐Order CFD Methods.

![](https://github.com/hujc91/pysif/blob/master/validation/enstrophy.png&s=200)

*Sif is the 
