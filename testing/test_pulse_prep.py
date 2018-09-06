import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose, assert_raises,assert_almost_equal
from overlaps import *

class Test_loss:
    def test_loss1(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = np.array([1, 1])
        int_fwm = sim_parameters(2.5e-20, 2, alphadB)
        int_fwm.general_options(1e-13, 1, 1, 1)
        int_fwm.propagation_parameters(10, 1000, 0.1)
        sim_wind = sim_window(fv, 1550e-9,1550e-9,int_fwm)

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
        int_fwm.propagation_parameters(10, 1000, 0.1)
        sim_wind = sim_window(fv, 1550e-9,1550e-9,int_fwm)

        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        maxim = np.max(alpha_func)
        assert maxim == 2*np.max(alphadB)/4.343




    def test_loss3(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = np.array([1, 2])
        int_fwm = sim_parameters(2.5e-20, 2, alphadB)
        int_fwm.general_options(1e-13, 1, 1, 1)
        int_fwm.propagation_parameters(10, 1000, 0.1)
        sim_wind = sim_window(fv, 1550e-9,1550e-9,int_fwm)

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
            self.N = 10
            self.nt = 2**self.N

    int_fwm = int_fwm1()
    lamp1 = 1549
    lamp2 = 1555
    lams = 1550
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    mins = np.min(1e-3*c/fv)
    f1 = 1e-3 * c / lamp1
    fs = fv[D_freq['where'][2]]
    f2 = 1e-3 * c / lamp2
    F = f1 - fs
    freqs = (f2 - F, f2, f2 + F, fs, f1, f1+F)
    assert_allclose(freqs, [fv[i] for i in D_freq['where']][::-1])
    fmin = fv.min()
    fmax = fv.max()
    assert np.all( [ i < fmax and i > fmin for i in freqs])

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
    index = 0
    master_index = 0
    M1, M2, Q_large = fibre_overlaps_loader(1)


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

        hf_exact = np.tile(hf_exact, (len(self.M2[1, :]), 1))
        assert_allclose(hf, hf_exact)


    def test_raman_analytic_1(self):
        ram = raman_object('on', 'analytic')
        D = loadmat('testing/testing_data/Raman_analytic.mat')
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
        t = D['t']
        t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
        dt = D['dt'][0][0]
        hf_exact = D['hf']
        hf_exact = np.asanyarray([hf_exact[i][0]
                                  for i in range(hf_exact.shape[0])])
        hf = ram.raman_load(t, dt, self.M2)
        assert_allclose(hf, hf_exact)


"----------------------------Dispersion operator--------------"


class Test_dispersion(Raman):


    int_fwm = sim_parameters(2.5e-20, 2, 0)
    int_fwm.general_options(1e-13, raman_object, 1, 'on')
    int_fwm.propagation_parameters(10, 10, 1)
    
    fv, D_freq = fv_creator(1549,1555, 1550, int_fwm)
    sim_wind = sim_window(fv, 1549e-9, 1.5508e-6, int_fwm)
    
    loss = Loss(int_fwm, sim_wind, amax=10)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB
    betas = load_disp_paramters(sim_wind.w0)
    Dop_large = dispersion_operator(betas, int_fwm, sim_wind)

    def test_dispersion_not_same(self):
        """
        Asserts that the dispersion of the two modes is not the same. 
        """
        assert_raises(AssertionError, assert_allclose,
                     self.Dop_large[0, :], self.Dop_large[1, :])
    


def test_betap():
    c_norm = c*1e-12
    lamda_c = 1.5508e-6
    w0 =  2 * pi * c / lamda_c
    

    betap1 = load_disp_paramters(w0,lamda_c)
    assert_allclose(betap1[0,:2], np.array([0,0]))


    D = np.array([19.8e6,21.8e6])
    S = np.array([0.068e15,0.063e15])

    beta2 = -D[:]*(lamda_c**2/(2*pi*c_norm))                                                #[ps**2/m]
    beta3 = lamda_c**4*S[:]/(4*(pi*c_norm)**2)+lamda_c**3*D[:]/(2*(pi*c_norm)**2)           #[ps**3/m]


    assert_allclose(betap1[0,:2], np.array([0,0])) 
    assert_allclose(betap1[1,1], -9.5e-02) 
    assert_allclose(betap1[:,2:], np.array([beta2, beta3]).T) 