import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose
from overlaps import *

def inputs(nm = 2):
    n2 = 2.5e-20                                # n2 for silica [m/W]
    # 0.0011666666666666668             # loss [dB/m]
    alphadB = np.array([0 for i in range(nm)])
    gama = 1e-3                                 # w/m
    "-----------------------------General options------------------------------"
    maxerr = 1e-13                # maximum tolerable error per step
    "----------------------------Simulation parameters-------------------------"
    N = 10
    z = 10                     # total distance [m]
    nplot = 10                  # number of plots
    nt = 2**N                     # number of grid points
    #dzstep = z/nplot            # distance per step
    dz_less = 1
    dz = 1         # starting guess value of the step

    lamp1 = 1549
    lamp2 = 1555
    lams = 1550
    lamda_c = 1.5508e-6
    lamda = lamp1*1e-9


    P_p1 = 1
    P_p2 = 1
    P_s = 1e-3
    return  n2, alphadB, gama, maxerr, N, z, nt,\
            lamp1, lamp2, lams, lamda_c, lamda, P_p1, P_p2, P_s, dz_less

def same_noise(nm = 2):
    
    n2, alphadB, gama, maxerr, N, z, nt,\
    lamp1, lamp2, lams, lamda_c, lamda,\
        P_p1, P_p2, P_s, dz_less = inputs(nm)

    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, 1, 'on')
    int_fwm.propagation_parameters(N, z, dz_less)
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm)
    noise_obj = Noise(int_fwm, sim_wind)
    return noise_obj

noise_obj = same_noise()




"-----------------------Pulse--------------------------------------------"



def pulse_propagations(ram, ss, nm, N_sol=1, cython = True, u = None):

    n2, alphadB, gama, maxerr, N, z, nt,\
        lamp1, lamp2, lams, lamda_c, lamda,\
         P_p1, P_p2, P_s, dz_less = inputs(nm)





    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z, dz_less)
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm)

    loss = Loss(int_fwm, sim_wind, amax=int_fwm.alphadB)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB
    dnerr = [0]
    index = 1
    master_index = 0
    a_vec = [2.2e-6]

    M1, M2, Q_large = fibre_overlaps_loader()
    betas = load_disp_paramters(sim_wind.w0)

    Dop = dispersion_operator(betas, int_fwm, sim_wind)

    integrand = Integrand(ram, ss, cython = cython, timing = False)
    dAdzmm = integrand.dAdzmm
    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')


    #M1, M2, Q = Q_matrixes(1, n2, lamda, gama=gama)
    raman = raman_object(int_fwm.ram, int_fwm.how)
    raman.raman_load(sim_wind.t, sim_wind.dt, M2)

    if raman.on == 'on':
        hf = raman.hf
    else:
        hf = None

    u = np.empty(
        [2, int_fwm.nm, len(sim_wind.t)], dtype='complex128')
    U = np.empty([2,int_fwm.nm,
                  len(sim_wind.t)], dtype='complex128')
    sim_wind.w_tiled = np.tile(sim_wind.w + sim_wind.woffset, (int_fwm.nm, 1))

    u[0,:, :] = noise_obj.noise


    
    
    woff1 = (D_freq['where'][1]+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0,0, :] += (P_p1)**0.5 * np.exp(1j*(woff1)*sim_wind.t)



    woff2 = (D_freq['where'][2]+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0,0, :] += (P_s)**0.5 * np.exp(1j*(woff2) *
                                           sim_wind.t)


    woff3 = (D_freq['where'][4]+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0,1, :] += (P_p2)**0.5 * np.exp(1j*(woff3) *
                                           sim_wind.t)



    U = fftshift(sim_wind.dt*fft(u), axes = -1)
    
    gam_no_aeff = -1j*int_fwm.n2*2*pi/sim_wind.lamda


    u[1,:,:], U[1,:,:] = pulse_propagation(u[0,:,:], U[0,:,:], int_fwm, M1, M2, Q_large,
                             sim_wind, hf, Dop, dAdzmm, gam_no_aeff)

    """
    fig1 = plt.figure()
    plt.plot(sim_wind.fv,w2dbm(np.abs(U[0,0,:])**2))
    plt.plot(sim_wind.fv,w2dbm(np.abs(U[0,1,:])**2))
    plt.savefig('1.png')
    plt.close()


    fig2 = plt.figure()
    plt.plot(sim_wind.fv,w2dbm(np.abs(U[1,0,:])**2))
    plt.plot(sim_wind.fv,w2dbm(np.abs(U[1,1,:])**2))
    plt.savefig('2.png')    
    plt.close()
    
    fig3 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[0,0,:])**2)
    plt.plot(sim_wind.t,np.abs(u[0,1,:])**2)
    plt.savefig('3.png')
    plt.close()


    fig4 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[1,0,:])**2)
    plt.plot(sim_wind.t,np.abs(u[1,1,:])**2)
    #plt.xlim(-10*T0, 10*T0)
    plt.savefig('4.png')    
    plt.close()

    fig5 = plt.figure()
    plt.plot(fftshift(sim_wind.w),(np.abs(U[1,0,:])**2 - np.abs(U[0,0,:])**2 ))
    plt.plot(fftshift(sim_wind.w),(np.abs(U[1,1,:])**2 - np.abs(U[0,1,:])**2 ))
    plt.savefig('error.png')
    plt.close()
    
    fig6 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[0,0,:])**2 - np.abs(u[1,0,:])**2)
    plt.plot(sim_wind.t,np.abs(u[0,1,:])**2 - np.abs(u[1,1,:])**2)
    plt.savefig('error2.png')
    plt.close()
    """
    return u, U, maxerr

"--------------------------------------------------------------------------"
class Test_cython(object):

    def test_ramoff_s0_nm2(self):
        u_c, U_c, maxerr = pulse_propagations('off', 0, nm=2, cython = True)
        u_p, U_p, maxerr = pulse_propagations('off', 0, nm=2, cython = False)
        a,b = np.sum(np.abs(u_c)**2), np.sum(np.abs(u_p)**2)
        assert np.allclose(a,b)

 
    def test_ramon_s0_nm2(self):
        u_c, U_c, maxerr = pulse_propagations('on', 0, nm=2, cython = True)
        u_p, U_p, maxerr = pulse_propagations('on', 0, nm=2, cython = False)
        a,b = np.sum(np.abs(u_c)**2), np.sum(np.abs(u_p)**2)
        assert np.allclose(a,b)
    
    def test_ramoff_s1_nm2(self):
        u_c, U_c, maxerr = pulse_propagations('off', 1, nm=2, cython = True)
        u_p, U_p, maxerr = pulse_propagations('off', 1, nm=2, cython = False)
        a,b = np.sum(np.abs(u_c)**2), np.sum(np.abs(u_p)**2)
        assert np.allclose(a,b)
    
    def test_ramon_s1_nm2(self):
        u_c, U_c, maxerr = pulse_propagations('on', 1, nm=2, cython = True)
        u_p, U_p, maxerr = pulse_propagations('on', 1, nm=2, cython = False)
        a,b = np.sum(np.abs(u_c)**2), np.sum(np.abs(u_p)**2)
        assert np.allclose(a,b)

    


class Test_pulse_prop_energy(object):

    def __test__(self,u):
        E = []
        for uu in u:
            sums = 0
            for umode in uu:
                print(umode.shape)
                sums += np.linalg.norm(umode)**2
            E.append(sums)
        np.allclose(E[0], E[1])

    def test_energy_r0_ss0_2(self):
        u, U, maxerr = pulse_propagations(
            'off', 0, nm=2, N_sol=np.abs(10*np.random.randn()))
        self.__test__(u)

    def test_energy_r0_ss1_2(self):
        u, U, maxerr = pulse_propagations(
            'off', 1, nm=2, N_sol=np.abs(10*np.random.randn()))
        self.__test__(u)

    def test_energy_r1_ss0_2(self):
        u, U, maxerr = pulse_propagations(
            'on', 0, nm=2, N_sol=np.abs(10*np.random.randn()))
        self.__test__(u)

    def test_energy_r1_ss1_2(self):
        u, U, maxerr = pulse_propagations(
            'on', 1, nm=2, N_sol=np.abs(10*np.random.randn()))
        self.__test__(u)


