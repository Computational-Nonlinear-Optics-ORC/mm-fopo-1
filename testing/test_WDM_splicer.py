import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, pi

class Test_splicer_1m():
    x1 = 950
    x2 = 1050
    N = 17
    nt = 2**N
    l1, l2 = 900, 1250
    f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

    fv = np.linspace(f1, f2, nt)
    lv = 1e3 * c / fv

    lamda = (lv[-1] + lv[0])/2
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-13, 'off', 1, 1)
    int_fwm.propagation_parameters(N, [0,18], 2, 1, 1)
    sim_wind = \
        sim_window(fv, lamda,
                   lamda, int_fwm, N)
    #sim_wind = sim_windows(lamda, lv, 900, 1250, nt)
    WDMS = WDM(x1, x2, fv, c)
    U1 = 10*(np.random.randn(1, nt) +
             1j * np.random.randn(1, nt))
    U2 = 10 * (np.random.randn(1, nt) +
               1j * np.random.randn(1, nt))
    splicer = Splicer(loss=np.random.rand()*10)
    U_in = (U1, U2)
    U1 = U1
    U2 = U2
    u_in1 = ifft(ifftshift(U1, axes = -1))
    u_in2 = ifft(ifftshift(U2, axes = -1))
    u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2
    U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
    a, b = splicer.pass_through(U_in, sim_wind)
    u_out1, u_out2 = a[0], b[0]
    U_out1, U_out2 = a[1], b[1]
    U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2
    u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

    def test1_splicer_freq(self):
        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test1_splicer_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot)


class Test_splicer_2m():
    x1 = 950
    x2 = 1050
    N = 15
    nt = 2**N
    l1, l2 = 900, 1250
    f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

    fv = np.linspace(f1, f2, nt)
    lv = 1e3 * c / fv

    lamda = (lv[-1] + lv[0])/2
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-13, 'off', 1, 1)
    int_fwm.propagation_parameters(N, [0,18], 2, 1, 1)
    sim_wind = \
        sim_window(fv, lamda,
                   lamda, int_fwm, N)
    #sim_wind = sim_windows(lamda, lv, 900, 1250, nt)
    WDMS = WDM(x1, x2, fv, c)
    U1 = 10*(np.random.randn(2, nt) +
             1j * np.random.randn(2, nt))
    U2 = 10*(np.random.randn(2, nt) +
             1j * np.random.randn(2, nt))
    splicer = Splicer(loss=np.random.rand()*10)
    U_in = (U1, U2)
    U1 = U1
    U2 = U2
    u_in1 = ifft(ifftshift(U1, axes = -1))
    u_in2 = ifft(ifftshift(U2, axes = -1))
    u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2
    U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
    a, b = splicer.pass_through(U_in, sim_wind)
    u_out1, u_out2 = a[0], b[0]
    U_out1, U_out2 = a[1], b[1]
    U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2
    u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

    def test2_splicer_freq(self):
        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test2_splicer_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot)




class Test_WDM_1m():
    x1 = 950
    x2 = 1050
    N = 18
    nt = 2**N
    l1, l2 = 900, 1250
    f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

    fv = np.linspace(f1, f2, nt)
    lv = 1e3 * c / fv

    lamda = (lv[-1] + lv[0])/2
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-13, 'off', 1, 1)
    int_fwm.propagation_parameters(N, [0,18], 2, 1, 1)
    sim_wind = \
        sim_window(fv, lamda,
                   lamda, int_fwm, N)
    #sim_wind = sim_windows(lamda, lv, 900, 1250, nt)
    WDMS = WDM(x1, x2, fv, c)

    U1 = 100*(np.random.randn(1, nt) +
              1j * np.random.randn(1, nt))
    U2 = 100 * (np.random.randn(1, nt) +
                1j * np.random.randn(1, nt))
    U_in = (U1, U2)
    U_in_tot = np.abs(U1)**2 + np.abs(U2)**2

    u_in1 = ifft(fftshift(U1, axes = -1))
    u_in2 = ifft(fftshift(U2, axes = -1))
    u_in_tot = simps(np.abs(u_in1)**2, sim_wind.t) + \
        simps(np.abs(u_in2)**2, sim_wind.t)

    a, b = WDMS.pass_through(U_in, sim_wind)

    U_out1, U_out2 = a[1], b[1]
    u_out1, u_out2 = a[0], b[0]

    U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

    u_out_tot = simps(np.abs(u_out1)**2, sim_wind.t) + \
        simps(np.abs(u_out2)**2, sim_wind.t)

    def test1m_WDM_freq(self):

        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test1m_WDM_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot, rtol=1e-05)


class Test_WDM_2m():
    """
    Tests conservation of energy in freequency and time space as well as the 
    absolute square value I cary around in the code.
    """
    x1 = 950
    x2 = 1050
    N = 18
    nt = 2**N
    l1, l2 = 900, 1250
    f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

    fv = np.linspace(f1, f2, nt)
    lv = 1e3 * c / fv

    lamda = (lv[-1] + lv[0])/2
    int_fwm = sim_parameters(2.5e-20, 2, 0)
    int_fwm.general_options(1e-13, 'off', 2, 2)
    int_fwm.propagation_parameters(N, [0,18], 2, 2, 1)
    sim_wind = \
        sim_window(fv, lamda,
                   lamda, int_fwm, N)
    #sim_wind = sim_windows(lamda, lv, 900, 1250, nt)
    WDMS = WDM(x1, x2, fv, c)
    U1 = 100*(np.random.randn(2, nt) +
              1j * np.random.randn(2, nt))
    U2 = 100*(np.random.randn(2, nt) +
              1j * np.random.randn(2, nt))
    U_in = (U1, U2)
    U_in_tot = np.abs(U1)**2 + np.abs(U2)**2

    u_in1 = ifft(fftshift(U1, axes = -1))
    u_in2 = ifft(fftshift(U2, axes = -1))
    u_in_tot = simps(np.abs(u_in1)**2, sim_wind.t) + \
        simps(np.abs(u_in2)**2, sim_wind.t)

    a, b = WDMS.pass_through(U_in, sim_wind)

    U_out1, U_out2 = a[1], b[1]
    u_out1, u_out2 = a[0], b[0]

    U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

    u_out_tot = simps(np.abs(u_out1)**2, sim_wind.t) + \
        simps(np.abs(u_out2)**2, sim_wind.t)

    def test2_WDM_freq(self):

        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test2_WDM_time(self):

        assert_allclose(self.u_in_tot, self.u_out_tot, rtol=1e-5)

def test_full_trans_in_cavity_1():
    N = 12
    nt = 2**N
    #fft,ifft,method = pick(nt, 1,100, 1)
    from scipy.constants import c, pi
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-6, raman_object, 0, 0)
    int_fwm.propagation_parameters(N, [0,18], 1, 1, 1)

    lam_p1 = 1048.17107345
    fv, where = fv_creator(lam_p1,1,0, 50, int_fwm)
    lv = 1e-3*c/fv
    sim_wind = sim_window(fv, lam_p1, lam_p1, int_fwm, 0)
    noise_obj = Noise(int_fwm, sim_wind)
    print(fv)
    WDM1 = WDM(1050, 1200, sim_wind.fv, c)
    WDM2 = WDM(930, 1200, sim_wind.fv, c)
    WDM3 = WDM(930, 1050, sim_wind.fv, c)
    WDM4 = WDM(930, 1200, sim_wind.fv, c)
    splicer1 = Splicer(loss=0.4895)
    splicer2 = Splicer(loss=0.142225011896)

    U = (1/2)**0.5 * (1 + 1j) * np.ones((1, nt))

    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM2.pass_through((U, np.zeros_like(U)), sim_wind)[1][1]

    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM1.pass_through((np.zeros_like(U), U), sim_wind)[0][1]

    assert_allclose(np.max(np.abs(U)**2), 0.7234722042243035, atol = 1e-2)


def test_full_trans_in_cavity_2():
    N = 12
    nt = 2**N
    #fft,ifft,method = pick(nt, 1,100, 1)
    from scipy.constants import c, pi
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-6, raman_object, 0, 0)
    int_fwm.propagation_parameters(N, [0,18], 1, 1, 1)

    lam_p1 = 1048.17107345
    fv, where = fv_creator(lam_p1,1,0, 50, int_fwm)
    lv = 1e-3*c/fv
    sim_wind = sim_window(fv, lam_p1, lam_p1, int_fwm, 0)
    noise_obj = Noise(int_fwm, sim_wind)
    print(fv)
    WDM1 = WDM(1050, 1200, sim_wind.fv, c)
    WDM2 = WDM(930, 1200, sim_wind.fv, c)
    WDM3 = WDM(930, 1050, sim_wind.fv, c)
    WDM4 = WDM(930, 1200, sim_wind.fv, c)
    splicer1 = Splicer(loss=0.4895)
    splicer2 = Splicer(loss=0.142225011896)

    U = (1/2)**0.5 * (1 + 1j) * np.ones((2, nt))

    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM2.pass_through((U, np.zeros_like(U)), sim_wind)[1][1]

    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM1.pass_through((np.zeros_like(U), U), sim_wind)[0][1]
    U_abs = np.abs(U)**2

    Umax1, Umax2 = np.max(U_abs[0, :]), np.max(U_abs[1, :])
    assert_allclose((Umax1, Umax2), (0.7234722042243035, 0.7234722042243035), atol = 1e-2)
