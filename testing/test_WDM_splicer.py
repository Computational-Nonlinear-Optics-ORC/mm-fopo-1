import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, pi


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
    fv = np.linspace(200, 600, 1024)
    alphadB = np.array([1, 1])
    int_fwm = sim_parameters(2.5e-20, 2, alphadB)
    int_fwm.general_options(1e-13, 1, 1, 1)
    int_fwm.propagation_parameters(10, 1000, 0.1)
    sim_wind = sim_window(fv, 1550e-9,1550e-9,int_fwm)
    #sim_wind = sim_windows(lamda, lv, 900, 1250, nt)
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

#(l1, l2, fv, fopa=False, with_resp = 'LP01'):



class Test_WDM_1m():
    x1 = 1549
    x2 = 1555
    N = 18
    nt = 2**N
    f1, f2 = 1e-3 * c / x1, 1e-3 * c / x2

    fv = np.linspace(f1, f2, nt)
    lv = 1e3 * c / fv

    lamda = (lv[-1] + lv[0])/2
    alphadB = np.array([1, 1])
    int_fwm = sim_parameters(2.5e-20, 2, alphadB)
    int_fwm.general_options(1e-13, 1, 1, 1)
    int_fwm.propagation_parameters(N, 1000, 0.1)
    sim_wind = sim_window(fv, 1550e-9,1550e-9,int_fwm)
    

    WDMS = WDM(x1, x2, fv, fopa = False, with_resp = 'LP01')

    U1 = 100*(np.random.randn(2, nt) +
              1j * np.random.randn(2, nt))
    U2 = 100 * (np.random.randn(2, nt) +
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


    def test1m_WDM_freq(self):
        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test1m_WDM_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot, rtol=1e-05)


class Test_perc_1m():
    lamp1 = 1549
    lamp2 = 1555
    lams = 1550
    N = 18
    nt = 2**N

    int_fwm = sim_parameters(2.5e-20, 2, [0,0])
    int_fwm.general_options(1e-15, 1, 1, 'on')
    int_fwm.propagation_parameters(N, 10, 100)
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamp1*1e-9, 1.5508e-6, int_fwm)
    

    perc = 100 * np.random.rand(len(D_freq['where']))

    print(perc)

    WDMS = Perc_WDM(D_freq['where'], perc, fv, 'false')


    U1 = 100*(np.random.randn(2, nt) +
              1j * np.random.randn(2, nt))
    U2 = 100 * (np.random.randn(2, nt) +
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


    def test1m_WDM_freq(self):
        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test1m_WDM_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot, rtol=1e-05)


class Test_bandpass_1m():
    lamp1 = 1549
    lamp2 = 1555
    lams = 1550
    N = 18
    nt = 2**N

    int_fwm = sim_parameters(2.5e-20, 2, [0,0])
    int_fwm.general_options(1e-15, 1, 1, 'on')
    int_fwm.propagation_parameters(N, 10, 100)
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamp1*1e-9, 1.5508e-6, int_fwm)
    

    perc = 100 * np.random.rand(len(D_freq['where']))

    print(perc)

    WDMS = Bandpass_WDM(D_freq['where'], perc, fv, 'false')


    U1 = 100*(np.random.randn(2, nt) +
              1j * np.random.randn(2, nt))
    U2 = 100 * (np.random.randn(2, nt) +
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


    def test1m_WDM_freq(self):
        assert_allclose(self.U_in_tot, self.U_out_tot)

    def test1m_WDM_time(self):
        assert_allclose(self.u_in_tot, self.u_out_tot, rtol=1e-05)



def test_WDM_phase_modulation():
    
    lamp1 = 1549
    lamp2 = 1555
    lams = 1550
    N = 10
    nt = 2**N

    int_fwm = sim_parameters(2.5e-20, 2, [0,0])
    int_fwm.general_options(1e-15, 1, 1, 'on')
    int_fwm.propagation_parameters(N, 10, 100)
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamp1*1e-9, 1.5508e-6, int_fwm)
    

    perc = 100 * np.random.rand(len(D_freq['where']))

    WDMS = Bandpass_WDM(D_freq['where'], perc, fv, 'false')


    pm = Phase_modulation_infase_WDM(D_freq['where'], WDMS)



    U1 = np.random.randint(0,100) * np.random.randn(2, nt) + \
         np.random.randint(0,100) * 1j * np.random.randn(2, nt)
    U2 = np.random.randint(0,100) * np.random.randn(2, nt) + \
         np.random.randint(0,100)  * 1j * np.random.randn(2, nt)
    
    U_in = (U1, U2)

    a_non = WDMS.pass_through(U_in, sim_wind)[0]
    U2cp = np.copy(U2)
    pm.modulate(U1,U2)
    assert_allclose(U2[1,:],U2cp[1,:])


    U_in = (U1, U2)

    a_mod= WDMS.pass_through(U_in, sim_wind)[0]
    
    sig_idx = D_freq['where'][2]


    e1 = np.abs(a_mod[1][0, sig_idx])**2
    e2 = np.abs(a_non[1][0, sig_idx])**2

    assert((e1 > e2) or np.allclose(e1,e2,equal_nan=True))
   