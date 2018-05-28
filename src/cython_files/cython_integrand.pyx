# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np


ctypedef double complex complex128_t
cdef double fr = 0.18
cdef double kr = 1 - fr

cpdef complex128_t[:, ::1] dAdzmm_ron_s1_cython(complex128_t[:, ::1] u0, complex128_t[:, ::1] u0_conj,
                                                              np.ndarray[long, ndim=2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:, ::1] Q,
                                                              double tsh, double dt, complex128_t[:, ::1] hf,
                                                              double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef long i, j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')
    cdef complex128_t[::1] Q_comp = np.empty(shapeM1, dtype='complex_')
    

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[:, ::1] M4 = fft(M3)

    for i in range(shapeM2):
        for j in range(shape2):
            M4[i, j] = M4[i, j] * hf[i, j]
    M4 = ifft(M4)
    M4 = fftshift(M4)


    dt = 3 * fr * dt 

    for i in range(shapeM1):
        Q_comp[i] = 2 * kr * Q[0, i] + kr * Q[1, i]

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q_comp[i] * M3[M1[4, i], j] +
                   dt*Q[0, i]*M4[M1[4, i], j])

    cdef complex128_t[:, ::1] M5 = fft(N)

    for i in range(shape1):
        for j in range(shape2):
            M5[i, j] = w_tiled[i, j] * M5[i, j]

    M5 = ifft(M5)

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * (N[i, j] + tsh * M5[i, j])
    return N


cpdef complex128_t[:, ::1] dAdzmm_ron_s0_cython(complex128_t[:, ::1] u0, const complex128_t[:, ::1] u0_conj,
                                                             np.ndarray[long, ndim= 2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:, ::1] Q,
                                                             double tsh, double dt, complex128_t[:, ::1] hf,
                                                             double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef long shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef long i, j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[:, ::1] M4 = fft(M3)
    for i in range(shapeM2):
        for j in range(shape2):
            M4[i, j] = M4[i, j] * hf[i, j]

    M4 = ifft(M4)
    M4 = fftshift(M4)

    cdef complex128_t[::1] Q_comp = np.empty(shapeM1, dtype='complex_')
    dt = 3 * fr * dt 

    for i in range(shapeM1):
        Q_comp[i] = 2 * kr * Q[0, i] + kr * Q[1, i]

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q_comp[i] * M3[M1[4, i], j] +
                   dt*Q[0, i]*M4[M1[4, i], j])
    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * N[i, j]
    return np.asarray(N)

cpdef complex128_t[:, ::1] dAdzmm_roff_s0_cython(complex128_t[:, ::1] u0, complex128_t[:, ::1] u0_conj,
                                                              np.ndarray[long, ndim= 2] M1, np.ndarray[long, ndim = 2] M2, complex128_t[:, ::1] Q,
                                                              double tsh, double dt, complex128_t[:, ::1] hf,
                                                              double[:, ::1] w_tiled, complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i, j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[::1] Q_comp = np.empty(shapeM1, dtype='complex_')

    for i in range(shapeM1):
        Q_comp[i] = 2 * kr * Q[0, i] + kr * Q[1, i]

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q_comp[i] * M3[M1[4, i], j])

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * N[i, j]
    return N


cpdef complex128_t[:, ::1] dAdzmm_roff_s1_cython(const complex128_t[:, ::1] u0, const complex128_t[:, ::1] u0_conj,
                                                              np.ndarray[long, ndim= 2] M1, np.ndarray[long, ndim = 2] M2, const complex128_t[:, ::1] Q,
                                                              const double tsh, double dt, const complex128_t[:, ::1] hf,
                                                              const double[:, ::1] w_tiled, const complex128_t gam_no_aeff):
    cdef int shape1 = u0.shape[0]
    cdef int shape2 = u0.shape[1]
    cdef int shapeM2 = M2.shape[1]
    cdef int shapeM1 = M1.shape[1]
    cdef int i, j
    cdef complex128_t[:, ::1] M3 = np.empty([shapeM2, shape2], dtype='complex_')
    cdef complex128_t[:, ::1] N = np.zeros([shape1, shape2], dtype='complex_')

    for i in range(shapeM2):
        for j in range(shape2):
            M3[i, j] = u0[M2[0, i], j]*u0_conj[M2[1, i], j]

    cdef complex128_t[::1] Q_comp = np.empty(shapeM1, dtype='complex_') 

    for i in range(shapeM1):
        Q_comp[i] = 2 * kr * Q[0, i] + kr * Q[1, i]

    for i in range(shapeM1):
        for j in range(shape2):
            N[M1[0, i], j] = N[M1[0, i], j] + u0[M1[1, i], j]\
                * (Q_comp[i] * M3[M1[4, i], j])

    cdef complex128_t[:, ::1] M5 = fft(N)

    for i in range(shape1):
        for j in range(shape2):
            M5[i, j] = w_tiled[i, j] * M5[i, j]

    M5 = ifft(M5)

    for i in range(shape1):
        for j in range(shape2):
            N[i, j] = gam_no_aeff * (N[i, j] + tsh * M5[i, j])
    return N

cdef complex128_t[:, ::1] fftshift(complex128_t[:, ::1] A):
    """
    FFTshift of memoryview written in Cython for Cython. Only
    works for even number of elements in the -1 axis. 
    """

    cdef int shape1  = A.shape[0]
    cdef int shape2  = A.shape[1]
    cdef int halfshape = shape2/2
    cdef int i,j,k
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


cdef np.ndarray[complex128_t, ndim= 2] fft(complex128_t[:, ::1] x_arr):
    """
    Uses MKL to perform 1D FFT on the input array x along the given axis.
    """
    cdef int shape = x_arr.shape[1]
    cdef np.ndarray[complex128_t, ndim = 2] x = np.asarray(x_arr)
    cdef np.ndarray[complex128_t, ndim= 2] f_arr =  __allocate_result(x, np.NPY_CDOUBLE)

    cdouble_cdouble_mkl_fft1d_out(x, shape, 1, f_arr)

    return f_arr

cdef np.ndarray[complex128_t, ndim= 2] ifft(complex128_t[:, ::1] x_arr):
    """
    Uses MKL to perform 1D iFFT on the input array x along the given axis.
    """
    cdef int shape = x_arr.shape[1]
    cdef np.ndarray[complex128_t, ndim = 2] x = np.asarray(x_arr)
    cdef np.ndarray[complex128_t, ndim = 2] f_arr =  __allocate_result(x, np.NPY_CDOUBLE)

    cdouble_cdouble_mkl_ifft1d_out(x, shape, 1, f_arr)
    return f_arr
