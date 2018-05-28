import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose, assert_raises,assert_almost_equal


class Test_loss:
    def test_loss1(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = np.array([1, 1])
        int_fwm = sim_parameters(2.5e-20, 2, alphadB)
        int_fwm.general_options(1e-13, 1, 1, 1)
        int_fwm.propagation_parameters(10, [0,18], 1, 100,10)
        sim_wind = sim_window(fv, 1,1, int_fwm, 1)

        loss = Loss(int_fwm, sim_wind, amax=alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        ex = np.zeros_like(alpha_func)
        for i, a in enumerate(alpha_func):
            ex[i, :] = np.ones_like(a)*alphadB[i]/4.343
        assert_allclose(alpha_func, ex)

    def test_loss2(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = np.array([1, 2])
        int_fwm = sim_parameters(2.5e-20, 2, alphadB)
        int_fwm.general_options(1e-13, 1, 1, 1)
        int_fwm.propagation_parameters(10, [0,18], 1, 100,10)
        sim_wind = sim_window(fv, 1,1, int_fwm, 1)

        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        maxim = np.max(alpha_func)
        assert maxim == 2*np.max(alphadB)/4.343




    def test_loss3(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = np.array([1, 2])
        int_fwm = sim_parameters(2.5e-20, 2, alphadB)
        int_fwm.general_options(1e-13, 1, 1, 1)
        int_fwm.propagation_parameters(10, [0,18], 1, 100,10)
        sim_wind = sim_window(fv, 1,1,int_fwm, 1)

        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        minim = np.min(alpha_func)
        assert minim == np.min(alphadB)/4.343




def test_fv_creator():
    """
    Checks whether the first order cascade is in the freequency window.
    """
    class int_fwm1(object):

        def __init__(self):
            self.N = 14
            self.nt = 2**self.N

    int_fwm = int_fwm1()
    lam_p1 = 1000
    lam_s = 1200
    #fv, where = fv_creator(lam_p1, lam_s, int_fwm)
    fv, where = fv_creator(lam_p1,lam_s,0, 50, int_fwm)
    mins = np.min(1e-3*c/fv)
    f1 = 1e-3*c/lam_p1
    fs = 1e-3*c/lam_s

    assert(all(i < max(fv) and i > min(fv)
               for i in (f1, fs)))


def test_noise():
    class sim_windows(object):

        def __init__(self):
            self.w = 10
            self.T = 0.1
            self.w0 = 9
    class int_fwms(object):

        def __init__(self):
            self.nt = 1024
            self.nm = 1
    int_fwm = int_fwms()
    sim_wind = sim_windows()
    noise = Noise(int_fwm, sim_wind)
    n1 = noise.noise_func(int_fwm)
    n2 = noise.noise_func(int_fwm)
    print(n1, n2)
    assert_raises(AssertionError, assert_almost_equal, n1, n2)


def test_time_frequency():
    nt = 3
    dt = np.abs(np.random.rand())*10
    u1 = 10*(np.random.randn(2**nt) + 1j * np.random.randn(2**nt))
    U = fftshift(dt*fft(u1))
    u2 = ifft(ifftshift(U)/dt)
    assert_allclose(u1, u2)



"----------------Raman response--------------"
#os.system('rm -r testing_data/step_index/*')


class Raman():
    l_vec = np.linspace(1600e-9, 1500e-9, 64)
    fv = 1e-12*c/l_vec
    dnerr = [0]
    index = 0
    master_index = 0
    a_vec = [2.2e-6]
    M1, M2, betas, Q_large = \
        fibre_parameter_loader(fv, a_vec, dnerr, index, master_index,
                               'step_index_2m', filepath='testing/testing_data/step_index/')



    def test_raman_off(self):
        ram = raman_object('off')
        ram.raman_load(np.random.rand(10), np.random.rand(1)[0], None)
        assert ram.hf == None


    def test_raman_load_1(self):
        ram = raman_object('on', 'load')
        #M1, M2, Q = Q_matrixes(1, 2.5e-20, 1.55e-6, 0.01)
        D = loadmat('testing/testing_data/Raman_measured.mat')
        t = D['t']
        t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
        dt = D['dt'][0][0]
        hf_exact = D['hf']
        hf_exact = np.asanyarray([hf_exact[i][0]
                                  for i in range(hf_exact.shape[0])])
        hf = ram.raman_load(t, dt, self.M2)

        #hf_exact = np.reshape(hf_exact, hf.shape)
        hf_exact = np.tile(hf_exact, (len(self.M2[1, :]), 1))
        assert_allclose(hf, hf_exact)


    def test_raman_analytic_1(self):
        ram = raman_object('on', 'analytic')
        D = loadmat('testing/testing_data/Raman_analytic.mat')
        #M1, M2, Q = Q_matrixes(1, 2.5e-20, 1.55e-6, 0.01)
        t = D['t']
        t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
        dt = D['dt'][0][0]
        hf_exact = D['hf']
        hf_exact = np.asanyarray([hf_exact[i][0]
                                  for i in range(hf_exact.shape[0])])
        hf = ram.raman_load(t, dt, self.M2)

        assert_allclose(hf, hf_exact)

    def test_raman_load_2(self):
        ram = raman_object('on', 'load')
        #M1, M2, Q = Q_matrixes(2, 2.5e-20, 1.55e-6, 0.01)
        D = loadmat('testing/testing_data/Raman_measured.mat')
        t = D['t']
        t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
        dt = D['dt'][0][0]
        hf_exact = D['hf']
        hf_exact = np.asanyarray([hf_exact[i][0]
                                  for i in range(hf_exact.shape[0])])
        hf = ram.raman_load(t, dt, self.M2)

        hf_exact = np.tile(hf_exact, (len(self.M2[1, :]), 1))
        assert_allclose(hf, hf_exact)

    def test_raman_analytic_2(self):
        ram = raman_object('on', 'analytic')
        D = loadmat('testing/testing_data/Raman_analytic.mat')
        #M1, M2, Q = Q_matrixes(2, 2.5e-20, 1.55e-6, 0.01)
        t = D['t']
        t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
        dt = D['dt'][0][0]
        hf_exact = D['hf']
        hf_exact = np.asanyarray([hf_exact[i][0]
                                  for i in range(hf_exact.shape[0])])
        hf = ram.raman_load(t, dt, self.M2)
        assert_allclose(hf, hf_exact)


"----------------------------Dispersion operator--------------"


class Test_dispersion_raman(Raman):

    l_vec = np.linspace(1600e-9, 1500e-9, 64)
    int_fwm = sim_parameters(2.5e-20, 2, 0)
    int_fwm.general_options(1e-13, 0, 1, 1)
    int_fwm.propagation_parameters(6, [0,18], 2, 1, 1)
    sim_wind = \
        sim_window(1e-12*c/l_vec, (l_vec[0]+l_vec[-1])*0.5,
                   (l_vec[0]+l_vec[-1])*0.5, int_fwm, 10)
    loss = Loss(int_fwm, sim_wind, amax=10)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB
    betas_disp = dispersion_operator(Raman.betas, int_fwm, sim_wind)

    def test_dispersion(self):
        """
        Compares the dispersion to a predetermined value.
        Not a very good test, make sure that the other one in this class
        passes. 
        """
      
        with h5py.File('testing/testing_data/betas_test1.hdf5', 'r') as f:
            betas_exact = f.get('betas').value

        assert_allclose(self.betas_disp, betas_exact)

    def test_dispersion_same(self):
        """
        Tests if the dispersion of the first two modes (degenerate) are the same. 
        """
        assert_allclose(self.betas_disp[:, 0, :], self.betas_disp[:, 1, :])


