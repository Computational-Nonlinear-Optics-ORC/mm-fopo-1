# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np

cdef extern from "complex.h":
    double complex cexp(double complex)
    double cabs(double complex)

ctypedef double complex complex128_t
ctypedef double double_t


############################################ Integrands ############################################
cpdef complex128_t[:, ::1] dAdzmm(complex128_t[:, ::1] u0,
                                            np.ndarray[unsigned int, ndim=2] M1, np.ndarray[unsigned int, ndim = 2] M2,
                                            double[:, ::1] Q, double tsh, complex128_t[:, ::1] hf,
                                            double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef long j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] u0_conj = np.conjugate(u0)

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[:, ::1] M4 = cyfft(M3)

    for i in range(shapeM2):
        for j in range(shape2):
            M4[i, j] = M4[i, j] * hf[i, j]
    M4 = cyifft(M4)
    M4 = cyfftshift(M4)


    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q[1,i] * M3[M1[4, i], j] +
                   Q[0, i]*M4[M1[4, i], j])
    cdef complex128_t[:, ::1] M5 = cyfft(N)

    for i in range(shape1):
        for j in range(shape2):
            M5[i, j] = w_tiled[i, j] * M5[i, j]

    M5 = cyifft(M5)

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * (N[i, j] + tsh * M5[i, j])

    return N


cpdef complex128_t[:, ::1] dAdzmm_ron_s0_cython(complex128_t[:, ::1] u0,
                                                              np.ndarray[unsigned int, ndim=2] M1, np.ndarray[unsigned int, ndim = 2] M2, double[:, ::1] Q,
                                                              double tsh, complex128_t[:, ::1] hf,
                                                              double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef long j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] u0_conj = np.conjugate(u0)
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[:, ::1] M4 = cyfft(M3)
    for i in range(shapeM2):
        for j in range(shape2):
            M4[i, j] = M4[i, j] * hf[i, j]

    M4 = cyifft(M4)
    M4 = cyfftshift(M4)


    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q[1, i] * M3[M1[4, i], j] +
                   Q[0, i] * M4[M1[4, i], j])
    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * N[i, j]
    return N


cpdef complex128_t[:, ::1] dAdzmm_roff_s0_cython(complex128_t[:, ::1] u0,
                                                              np.ndarray[unsigned int, ndim=2] M1, np.ndarray[unsigned int, ndim = 2] M2, double[:, ::1] Q,
                                                              double tsh, complex128_t[:, ::1] hf,
                                                              double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef long j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] u0_conj = np.conjugate(u0)
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]


    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q[1,i] * M3[M1[4, i], j])

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * N[i, j]
    return N


cpdef complex128_t[:, ::1] dAdzmm_roff_s1_cython(complex128_t[:, ::1] u0,
                                                              np.ndarray[unsigned int, ndim=2] M1, np.ndarray[unsigned int, ndim = 2] M2, double[:, ::1] Q,
                                                              double tsh, complex128_t[:, ::1] hf,
                                                              double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i
    cdef long j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] u0_conj = np.conjugate(u0)
    
    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]



    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q[1,i] * M3[M1[4, i], j])

    cdef complex128_t[:, ::1] M5 = cyfft(N)

    for i in range(shape1):
        for j in range(shape2):
            M5[i, j] = w_tiled[i, j] * M5[i, j]

    M5 = cyifft(M5)

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * (N[i, j] + tsh * M5[i, j])
    return N
############################################ Pulse_prop ############################################

DEF Safety = 0.95
cpdef pulse_propagation(complex128_t[:,::1] u1, double dz, double dzstep, double maxerr,
                        np.ndarray[unsigned int, ndim=2] M1, np.ndarray[unsigned int, ndim=2] M2,
                        double[:, ::1] Q, double[:, ::1] w_tiled,double tsh, complex128_t[:, ::1] hf,
                        complex128_t[:, ::1] Dop,  complex128_t gam_no_aeff):
                        

    """Pulse propagation using SSFM"""

    cdef double dztot = 0.  # total distance traveled
    cdef int shape1 = u1.shape[0]
    cdef long shape2 = u1.shape[1]

    cdef complex128_t[:,::1] u1new = np.empty([shape1,shape2], dtype='complex_')
    cdef complex128_t[:,::1] U = np.empty([shape1,shape2], dtype='complex_')
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    cdef double_t[::1] delta = np.empty(1, dtype = 'double')
    cdef int exitt = 0
    cdef double  temp, temp2


    while exitt == 0:
        # trick to do the first iteration
        delta[0] = 2*maxerr
        while delta[0] > maxerr:
            u1new = half_disp_step(u1, Dop, dz, shape1, shape2)
            A = RK45CK(delta, u1new, M1, M2, Q, tsh, hf,
                    w_tiled, gam_no_aeff, dz, shape1, shape2)
            if (delta[0] > maxerr):
                # calculate the step (shorter) to redo
                dz = dz * Safety*(maxerr/delta[0])**0.25


        #############Successful step###############
        u1 = half_disp_step(A, Dop, dz, shape1,shape2)

        dztot = dztot + dz

        if (delta[0] == 0):
            dz = Safety*dzstep
        else:
            temp = Safety*dz*(maxerr/delta[0])**0.2
            temp2 = Safety*dzstep
            dz = min(temp, temp2)
        ###################################################################
        temp = dztot + dz
        if (dztot == dzstep):
            exitt = 1
        elif (temp >= dzstep):
            dz = dzstep - dztot
        ###################################################################

    U = cyfft(u1)
    U = cyfftshift(U)
    return np.asarray(U), dz


cpdef complex128_t[:,::1] half_disp_step(complex128_t[:,::1] u,
                         complex128_t[:,::1] Dop, double dz, int shape1, long shape2):
    #np.fft.ifft(np.exp(Dop*dz/2) * np.fft.fft(u1))
    cdef complex128_t[:,::1] temp1 = np.empty([shape1,shape2], dtype='complex_')
    cdef complex128_t[:,::1] temp2 = np.empty([shape1,shape2], dtype='complex_')
    cdef int i
    cdef long j
    for i in range(shape1):
        for j in range(shape2):
            temp1[i,j] = cexp(Dop[i,j] * dz)
    temp2 = cyfft(u)
    for i in range(shape1):
        for j in range(shape2):
            temp2[i,j] = temp1[i,j] * temp2[i,j]
    temp2 = cyifft(temp2)
    return temp2

############################################ RKCash ############################################

cdef complex128_t[:,::1] RK45CK(double_t[::1] delta, complex128_t[:, ::1] u1, np.ndarray[unsigned int, ndim=2] M1,
                                np.ndarray[unsigned int, ndim = 2] M2, double[:, ::1] Q, double tsh,
                                complex128_t[:, ::1] hf,double[:, ::1] w_tiled, complex128_t gam_no_aeff, double dz,
                                int  shape1, long shape2):
    """
    Propagates the nonlinear operator for 1 step using a 5th order Runge
    Kutta method
    use: [A delta] = RK5mm(u1, dz)
    where u1 is the initial time vector
    hf is the Fourier transform of the Raman nonlinear response time
    dz is the step over which to propagate

    in output: A is new time vector
    delta is the norm of the maximum estimated error between a 5th
    order and a 4th order integration
    """
    cdef complex128_t gama_temp = dz * gam_no_aeff



    cdef complex128_t[:,::1] A1 = dAdzmm(u1, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] u2 = A2_temp(u1, A1, shape1,shape2)

    cdef complex128_t[:,::1] A2 = dAdzmm(u2, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] u3 = A3_temp(u1, A1, A2, shape1,shape2)

    cdef complex128_t[:,::1] A3 = dAdzmm(u3, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] u4 = A4_temp(u1, A1, A2, A3, shape1,shape2)

    cdef complex128_t[:,::1] A4 = dAdzmm(u4, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] u5 = A5_temp(u1, A1, A2, A3, A4, shape1,shape2)

    cdef complex128_t[:,::1] A5 = dAdzmm(u5, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] u6 = A6_temp(u1, A1, A2, A3, A4, A5, shape1,shape2)

    cdef complex128_t[:,::1] A6 = dAdzmm(u6, M1, M2, Q, tsh, hf, w_tiled,gama_temp)

    cdef complex128_t[:,::1] A = A_temp(u1, A1, A3, A4, A6, shape1,shape2)  # Fifth order accuracy

    cdef complex128_t[:,::1] Afourth = Afourth_temp(u1, A1, A3, A4, A5, A6, A, shape1,shape2)  # Fourth order accuracy

    delta[0] = norm(Afourth, shape1,shape2)


    return A


cpdef complex128_t[:,::1] A2_temp(complex128_t[:,::1] u1, complex128_t[:,::1] A1, int shape1, long shape2):
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    cdef int i
    cdef long j
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] = u1[i,j] + (1./5)*A1[i,j]
    return A


cpdef complex128_t[:,::1] A3_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                    complex128_t[:,::1] A2, int shape1, long shape2):

    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] =  u1[i,j] + (3./40)*A1[i,j] + (9./40)*A2[i,j]
    return A



cpdef complex128_t[:,::1] A4_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                    complex128_t[:,::1] A2,complex128_t[:,::1] A3,
                                    int shape1, long shape2):
    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] = u1[i,j] + (3./10)*A1[i,j] - (9./10)*A2[i,j] + (6./5)*A3[i,j]
    return A


cpdef complex128_t[:,::1] A5_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                    complex128_t[:,::1] A2,complex128_t[:,::1] A3,
                                    complex128_t[:,::1] A4, int shape1, long shape2):

    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] = u1[i,j] - (11./54)*A1[i,j] + (5./2)*A2[i,j] - (70./27)*A3[i,j] + (35./27)*A4[i,j]
    return A



cpdef complex128_t[:,::1] A6_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                complex128_t[:,::1] A2,complex128_t[:,::1] A3,complex128_t[:,::1] A4,
                                complex128_t[:,::1] A5, int shape1, long shape2):
    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] =  u1[i,j] + (1631./55296)*A1[i,j] + (175./512)*A2[i,j]+ (575./13824)*A3[i,j] +\
                        (44275./110592)*A4[i,j] + (253./4096)*A5[i,j]
    return A


cpdef complex128_t[:,::1] Afourth_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                        complex128_t[:,::1] A3,complex128_t[:,::1] A4,
                                        complex128_t[:,::1] A5,complex128_t[:,::1] A6,
                                        complex128_t[:,::1] Afifth, int shape1, long shape2):
    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    cdef complex128_t[:,::1] Aerr = np.empty([shape1,shape2], dtype='complex_')
    
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] = u1[i,j] + (2825./27648)*A1[i,j] + (18575./48384)*A3[i,j] + (13525./55296) * \
        A4[i,j] + (277./14336)*A5[i,j] + (1./4)*A6[i,j]
    
    for i in range(shape1):
        for j in range(shape2):
            Aerr[i,j] = Afifth[i,j] - A[i,j]
    return Aerr


cpdef complex128_t[:,::1] A_temp(complex128_t[:,::1] u1,complex128_t[:,::1] A1,
                                 complex128_t[:,::1] A3,complex128_t[:,::1] A4,
                                 complex128_t[:,::1] A6, int shape1, long shape2):
    cdef int i
    cdef long j
    cdef complex128_t[:,::1] A = np.empty([shape1,shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A[i,j] = u1[i,j] + (37./378)*A1[i,j] + (250./621)*A3[i,j] + (125./594) * \
                        A4[i,j] + (512./1771)*A6[i,j]
    return A
############################################ Suplementary ############################################

cpdef double_t norm(complex128_t[:,::1] A, int shape1, long shape2):

    cdef int i
    cdef long j
    cdef double[::1] sum1 = np.zeros(shape1, dtype='double')
    cdef complex128_t[:,::1] A_2 = np.empty([shape1, shape2], dtype='complex_')
    for i in range(shape1):
        for j in range(shape2):
            A_2[i,j] = cabs(A[i,j])**2
    for i in range(shape1):
        for j in range(shape2):
            sum1[i] = sum1[i] + <double>(A_2[i,j])
    for i in range(shape1):
        sum1[i] = sum1[i]**0.5
    cdef double res = max(sum1[0], sum1[1])

    return res


cpdef complex128_t[:, ::1] cyfftshift(complex128_t[:, ::1] A):
    """
    FFTshift of memoryview written in Cython for Cython. Only
    works for even number of elements in the -1 axis. 
    """
    cdef int shape1 = A.shape[0]
    cdef long shape2 = A.shape[1]
    cdef long halfshape = shape2/2
    cdef int i
    cdef long j,k
    cdef complex128_t[:, ::1] B = np.empty([shape1,shape2], dtype = 'complex_')
    

    for i in range(shape1):
        k = 0
        for j in range(halfshape, shape2):
            B[i,k] = A[i,j]
            k = k +1
        for j in range(halfshape):
            B[i,k] = A[i,j]
            k = k +1
    return B








########################Intel-MKL part##############################
# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from libc.string cimport memcpy

cdef extern from "Python.h":
    ctypedef int size_t

    void * PyMem_Malloc(size_t n)
    void PyMem_Free(void * buf)


# These are commented out in the numpy support we cimported above.
# Here I have declared them as taking void* instead of PyArrayDescr
# and object. In this file, only NULL is passed to these parameters.


cdef extern from "src/mklfft.h":
    int cdouble_cdouble_mkl_fft1d_out(np.ndarray, int, int, np.ndarray)
    int cdouble_cdouble_mkl_ifft1d_out(np.ndarray, int, int, np.ndarray)


# Initialize numpy
np.import_array()


cdef np.ndarray __allocate_result(np.ndarray x_arr, int f_type):
    """
    An internal utility to allocate an empty array for output of not-in-place FFT.
    """
    cdef np.npy_intp * f_shape
    cdef np.ndarray f_arr

    f_ndim = np.PyArray_NDIM(x_arr)

    f_shape = <np.npy_intp*> PyMem_Malloc(f_ndim * sizeof(np.npy_intp))
    memcpy(f_shape, np.PyArray_DIMS(x_arr), f_ndim * sizeof(np.npy_intp))

    # allocating output buffer
    f_arr = <np.ndarray > np.PyArray_EMPTY(
        f_ndim, f_shape, < np.NPY_TYPES > f_type, 0)  # 0 for C-contiguous
    PyMem_Free(f_shape)

    return f_arr


cpdef np.ndarray[complex128_t, ndim= 2] cyfft(complex128_t[:, ::1] x_arr):
    """
    Uses MKL to perform 1D FFT on the input array x along the given axis.
    """
    cdef shape = x_arr.shape[1]
    cdef np.ndarray[complex128_t, ndim = 2] x = np.asarray(x_arr)
    cdef np.ndarray[complex128_t, ndim= 2] f_arr =  __allocate_result(x, np.NPY_CDOUBLE)

    cdouble_cdouble_mkl_fft1d_out(x, shape, 1, f_arr)

    return f_arr

cpdef np.ndarray[complex128_t, ndim= 2] cyifft(complex128_t[:, ::1] x_arr):
    """
    Uses MKL to perform 1D iFFT on the input array x along the given axis.
    """
    cdef shape = x_arr.shape[1]
    cdef np.ndarray[complex128_t, ndim = 2] x = np.asarray(x_arr)
    cdef np.ndarray[complex128_t, ndim = 2] f_arr =  __allocate_result(x, np.NPY_CDOUBLE)

    cdouble_cdouble_mkl_ifft1d_out(x, shape, 1, f_arr)
    return f_arr
