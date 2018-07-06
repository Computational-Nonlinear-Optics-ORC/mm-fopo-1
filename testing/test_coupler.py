import sys
sys.path.append('src')
import coupler
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, pi
import os
from scipy import special

def test_bessel():
    assert_allclose(coupler.jv_(0,0), 0.)
    assert np.isnan(coupler.kv_(0,0))

class Test_eigensolver:
    e = coupler.Eigensolver(1549e-9, 1555e-9, 10)
    e.initialise_fibre(5e-6)


    def test_core_clad(self):
    	assert (self.e.ncore > self.e.nclad).all()
    def test_V_number_eigen(self):

        u_01, w_01, neff01 = [np.zeros([1, 10]) for i in range(3)]
        u_11, w_11, neff11 = [np.zeros([2, 10]) for i in range(3)]
        for i in range(10):
            u_01[0, i], w_01[0, i], neff01[0, i] = self.e.solve_01(i)
            u_11[:, i], w_11[:, i], neff11[:, i] = self.e.solve_11(i)

        assert_allclose((u_01[0,:]**2 + w_01[0,:]**2)**0.5, self.e.V_vec)
        assert_allclose((u_11[0,:]**2 + w_11[0,:]**2)**0.5, self.e.V_vec)
        assert_allclose((u_11[1,:]**2 + w_11[1,:]**2)**0.5, self.e.V_vec)
        assert (neff01[0,:] > neff11[0, :]).all()
        assert (neff01[0,:] > neff11[1, :]).all()

class Test_coupling:
    lmin = 1540e-9
    lmax = 1560e-9
    N_l = 10
    a = 5e-6
    N_points = 128
    d_vec = [1.1e-6, 1.95e-6]
    k01_1, k11_1, couple01_1, couple11_1, \
        k01_2, k11_2, couple01_2, couple11_2 = \
        coupler.calculate_coupling_coeff(lmin, lmax, a, N_points, N_l, d_vec)

    def test_coupling_less(self):
        assert (self.k01_1 < self.k11_1).all()
        assert (self.k01_2 < self.k11_2).all()
        assert (self.k01_1 > self.k01_2).all()
        assert (self.k11_1 > self.k11_2).all()