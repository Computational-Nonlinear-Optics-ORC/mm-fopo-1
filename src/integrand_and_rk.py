import numpy as np
from scipy.constants import pi
from six.moves import builtins
from numpy.fft import fftshift
from scipy.fftpack import fft, ifft
try:
    from cython_files.cython_integrand import *
except ModuleNotFoundError:


    print('Warning, cython was not able to complile')
    pass
import sys
assert_allclose = np.testing.assert_allclose
import numba
complex128 = numba.complex128

vectorize = numba.vectorize
autojit, jit = numba.autojit, numba.jit
cfunc = numba.cfunc
generated_jit = numba.generated_jit
guvectorize = numba.guvectorize

# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func
from time import time
import pickle


trgt = 'cpu'
#trgt = 'parallel'
#trgt = 'cuda'



#@jit(nogil = True)
def dAdzmm_roff_s0(u0,u0_conj, M1, M2, Q, tsh, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,u0_conj,M2)
    N = nonlin_kerr(M1, Q, u0, M3)
    N *= gam_no_aeff
    return N



#@jit(nogil = True)
def dAdzmm_roff_s1(u0,u0_conj, M1, M2, Q, tsh, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,u0_conj,M2)
    N = nonlin_kerr(M1, Q, u0, M3)
    N = gam_no_aeff * (N + tsh*ifft(w_tiled * fft(N)))
    return N


def dAdzmm_ron_s0(u0,u0_conj, M1, M2, Q, tsh, hf, w_tiled, gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,u0_conj,M2)
    M4 = fftshift(ifft(fft(M3)*hf), axes = -1) # creates matrix M4
    N = nonlin_ram(M1, Q, u0, M3, M4)
    N *= gam_no_aeff
    return N



def dAdzmm_ron_s1(u0,u0_conj, M1, M2, Q, tsh, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,u0_conj,M2)
    M4 = fftshift(ifft(multi(fft(M3),hf)), axes = -1) # creates matrix M4
    N = nonlin_ram(M1, Q, u0, M3, M4)

    N = gam_no_aeff * (N + tsh*ifft(multi(w_tiled,fft(N))))
    return N


@jit(nopython=True,nogil = True)
def multi(a,b):
    return a * b


@guvectorize(['void(complex128[:,:],complex128[:,:], int64[:,:], complex128[:,:])'],\
                 '(n,m),(n,m),(o,l)->(l,m)',target = trgt)
def uabs(u0,u0_conj,M2,M3):
    for ii in range(M2.shape[1]):
        M3[ii,:] = u0[M2[0,ii],:]*u0_conj[M2[1,ii],:]


@guvectorize(['void(int64[:,:], complex128[:,:], complex128[:,:],\
            complex128[:,:], complex128[:,:], complex128[:,:])'],\
            '(w,a),(i,a),(m,n),(l,n),(l,n)->(m,n)',target = trgt)
def nonlin_ram(M1, Q, u0, M3, M4, N):
    N[:,:] = 0
    for ii in range(M1.shape[1]):
        N[M1[0,ii],:] += u0[M1[1,ii],:]*( Q[1,ii] \
                                *M3[M1[4,ii],:] + \
                               Q[0,ii]*M4[M1[4,ii],:])


@guvectorize(['void(int64[:,:], complex128[:,:], complex128[:,:],\
            complex128[:,:], complex128[:,:])'],\
            '(w,a),(i,a),(m,n),(l,n)->(m,n)',target = trgt)
def nonlin_kerr(M1, Q, u0, M3, N):
    N[:,:] = 0
    for ii in range(M1.shape[1]):
        N[M1[0,ii],:] += (Q[1,ii]) \
                                *u0[M1[1,ii],:]*M3[M1[4,ii],:]
                

@vectorize(['complex128(float64,float64,complex128,\
			float64,complex128,float64)'], target=trgt)
def self_step(n2, lamda, N, tsh, temp, rp):
    return -1j*n2*2*rp/lamda*(N + tsh*temp)



class Integrand(object):
    def __init__(self,ram, ss, cython = True, timing = False):
        if cython:
            if ss == 0 and ram == 'off':
                self.dAdzmm = dAdzmm_roff_s0_cython
            elif ss == 0 and ram == 'on':
                self.dAdzmm = dAdzmm_ron_s0_cython
            elif ss == 1 and ram == 'off':
                self.dAdzmm = dAdzmm_roff_s1_cython
            else:
                self.dAdzmm = dAdzmm
        else:
            if ss == 0 and ram == 'off':
                self.dAdzmm = dAdzmm_roff_s0
            elif ss == 0 and ram == 'on':
                self.dAdzmm = dAdzmm_ron_s0
            elif ss == 1 and ram == 'off':
                self.dAdzmm = dAdzmm_roff_s1
            else:
                self.dAdzmm = dAdzmm_ron_s1
        if timing:
            self.dAdzmm = self.timer

    def timer(self,u0,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff):
        """
        Times the functions of python, cython etc. 
        """
        dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8 = [], [], [], [],\
                                                [], [], [], []

        NN = 100
        for i in range(NN):
            '------No ram, no ss--------'
            t = time()
            N1 = dAdzmm_roff_s0_cython(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt1.append(time() - t)

            t = time()
            N2 = dAdzmm_roff_s0(u0,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt2.append(time() - t)
            assert_allclose(N1, N2)
            
            '------ ram, no ss--------'
            t = time()
            N1 = dAdzmm_ron_s0_cython(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt3.append(time() - t)

            t = time()
            N2 = dAdzmm_ron_s0(u0,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt4.append(time() - t)
            assert_allclose(N1, N2)


            '------ no ram, ss--------'
            t = time()
            N1 = dAdzmm_roff_s1_cython(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt5.append(time() - t)

            t = time()
            N2 = dAdzmm_roff_s1(u0,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt6.append(time() - t)
            assert_allclose(N1, N2)
            
            '------ ram, ss--------'
            t = time()
            N1 = dAdzmm(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt7.append(time() - t)

            t = time()
            N2 = dAdzmm_ron_s1(u0,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
            dt8.append(time() - t)
            assert_allclose(N1, N2)
            
        print('cython_ram(off)_s0: {} +/- {}'.format(np.average(dt1),np.std(dt1)))
        print('python_ram(off)_s0: {} +/- {}'.format(np.average(dt2),np.std(dt2)))
        print('Cython is {} times faster'.format(np.average(dt2)/np.average(dt1)))
        print('--------------------------------------------------------')
        print('cython_ram(on)_s0: {} +/- {}'.format(np.average(dt3),np.std(dt3)))
        print('python_ram(on)_s0: {} +/- {}'.format(np.average(dt4),np.std(dt4)))
        print('Cython is {} times faster'.format(np.average(dt4)/np.average(dt3)))
        print('--------------------------------------------------------')

        print('cython_ram(off)_s1: {} +/- {}'.format(np.average(dt5),np.std(dt5)))
        print('python_ram(off)_s1: {} +/- {}'.format(np.average(dt6),np.std(dt6)))
        print('Cython is {} times faster'.format(np.average(dt6)/np.average(dt5)))
        print('--------------------------------------------------------')

        print('cython_ram(on)_s1: {} +/- {}'.format(np.average(dt7),np.std(dt7)))
        print('python_ram(on)_s1: {} +/- {}'.format(np.average(dt8),np.std(dt8)))
        print('Cython is {} times faster'.format(np.average(dt8)/np.average(dt7)))
        print('--------------------------------------------------------')
        sys.exit()
        return N

