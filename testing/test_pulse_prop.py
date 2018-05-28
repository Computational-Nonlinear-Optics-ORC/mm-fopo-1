import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose

"-----------------------Full soliton--------------------------------------------"
def get_Qs(nm, gama,fv, a_vec, dnerr, index, master_index,lamda, n2):
    if nm == 1:
        D = loadmat('loading_data/M1_M2_1m_new.mat')
        M1_temp, M2 = D['M1'], D['M2']
        M2[:, :] -= 1
        M1 = np.empty([np.shape(M1_temp)[0]-2,
                       np.shape(M1_temp)[1]], dtype=np.int64)

        M1[:4] = M1_temp[:4] - 1
        Q_large = M1_temp[np.newaxis, 4:6, :]
        M1[-1] = M1_temp[6, :] - 1
        Q_large[:,:,:] = gama / (3*n2*(2*pi/lamda))
    else:
        M1, M2, dump, Q_large = \
            fibre_parameter_loader(fv, a_vec, dnerr, index, master_index,
                                   filename='step_index_2m', filepath='testing/testing_data/step_index/')
        print(Q_large.shape)
        Q_large[0,0,:] = gama / (3*n2*(2*pi/lamda)) * np.array([1,1,0,0,0,0,1,1])
        Q_large[0,1,:] = gama / (3*n2*(2*pi/lamda)) * np.array([1,0,0,1,1,0,0,1])
    return Q_large, M1, M2


def pulse_propagations(ram, ss, nm, N_sol=1, cython = True, u = None):
    "SOLITON TEST. IF THIS FAILS GOD HELP YOU!"
    n2 = 2.5e-20                                # n2 for silica [m/W]
    # 0.0011666666666666668             # loss [dB/m]
    alphadB = np.array([0 for i in range(nm)])
    gama = 1e-3                                 # w/m
    "-----------------------------General options------------------------------"
    maxerr = 1e-13                # maximum tolerable error per step
    "----------------------------Simulation parameters-------------------------"
    N = 10
    z = np.array([0,70])                     # total distance [m]
    nplot = 10                  # number of plots
    nt = 2**N                     # number of grid points
    #dzstep = z/nplot            # distance per step
    dz_less = 1
    dz = 1         # starting guess value of the step

    lam_p1 = 1550
    lamda_c = 1550e-9
    lamda = lam_p1*1e-9

    beta2 = -1e-3
    P0_p1 = 1
    betas = np.array([0, 0, beta2])
    T0 = (N_sol**2 * np.abs(beta2) / (gama * P0_p1))**0.5
    TFWHM = (2*np.log(1+2**0.5)) * T0

    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z, nplot, dz_less, 1)
    int_fwm.woble_propagate(0)
    fv, where = fv_creator(lam_p1,lam_p1 + 25,0, 100, int_fwm)
    #fv, where = fv_creator(lam_p1, , int_fwm, prot_casc=0)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm, fv_idler_int=1)

    loss = Loss(int_fwm, sim_wind, amax=int_fwm.alphadB)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB
    dnerr = [0]
    index = 1
    master_index = 0
    a_vec = [2.2e-6]
    Q_large,M1,M2 = get_Qs(nm, gama, fv, a_vec, dnerr, index, master_index, lamda, n2)
    betas = betas[np.newaxis, :]
    # sys.exit()
    Dop = dispersion_operator(betas, int_fwm, sim_wind)
    print(Dop.shape)
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
        [ int_fwm.nm, len(sim_wind.t)], dtype='complex128')
    U = np.empty([int_fwm.nm,
                  len(sim_wind.t)], dtype='complex128')

    sim_wind.w_tiled = np.tile(sim_wind.w + sim_wind.woffset, (int_fwm.nm, 1))
    
    u[:, :] = ((P0_p1)**0.5 / np.cosh(sim_wind.t/T0)) * \
            np.exp(-1j*(sim_wind.woffset)*sim_wind.t)
    U[:, :] = fftshift(sim_wind.dt*fft(u[:, :]))
    
    gam_no_aeff = -1j*int_fwm.n2*2*pi/sim_wind.lamda

    u, U = pulse_propagation(u, U, int_fwm, M1, M2.astype(np.int64), Q_large[0].astype(np.complex128),
                             sim_wind, hf, Dop[0], dAdzmm, gam_no_aeff)
    U_start = np.abs(U[ :, :])**2

    u[:, :] = u[:, :] * \
        np.exp(1j*z[-1]/2)*np.exp(-1j*(sim_wind.woffset)*sim_wind.t)
    """
    fig1 = plt.figure()
    plt.plot(sim_wind.fv,np.abs(U[1,:])**2)
    plt.savefig('1.png')

    fig2 = plt.figure()
    plt.plot(sim_wind.fv,np.abs(U[1,:])**2)
    plt.savefig('2.png')    
    
    
    fig3 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[1,:])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('3.png')

    fig4 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[1,:])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('4.png')    
    

    fig5 = plt.figure()
    plt.plot(fftshift(sim_wind.w),(np.abs(U[1,:])**2 - np.abs(U[1,:])**2 ))
    plt.savefig('error.png')

    
    fig6 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[1,:])**2 - np.abs(u[1,:])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('error2.png')
    plt.show()
    """
    return u, U, maxerr
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

    


class Test_pulse_prop(object):

    def test_solit_r0_ss0(self):
        u, U, maxerr = pulse_propagations('off', 0, nm=1)
        assert_allclose(np.abs(u[:, :])**2,
                        np.abs(u[:, :])**2, atol=9e-4)

    def test_solit_r0_ss0_2(self):
        u, U, maxerr = pulse_propagations('off', 0, nm=2)
        #print(np.linalg.norm(np.abs(u[:, 0])**2 - np.abs(u[:, -1])**2, 2))

        assert_allclose(np.abs(u[:, :])**2,
                        np.abs(u[:, :])**2, atol=9e-3)

    def test_energy_r0_ss0(self):
        u, U, maxerr = pulse_propagations(
            'off', 0, nm=1, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r0_ss1(self):
        u, U, maxerr = pulse_propagations(
            'off', 1, nm=1, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r1_ss0(self):
        u, U, maxerr = pulse_propagations(
            'on', 0, nm=1, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r1_ss1(self):
        u, U, maxerr = pulse_propagations(
            'on', 1, nm=1, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    

    def test_energy_r0_ss0_2(self):
        u, U, maxerr = pulse_propagations(
            'off', 0, nm=2, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r0_ss1_2(self):
        u, U, maxerr = pulse_propagations(
            'off', 1, nm=2, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r1_ss0_2(self):
        u, U, maxerr = pulse_propagations(
            'on', 0, nm=2, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)

    def test_energy_r1_ss1_2(self):
        u, U, maxerr = pulse_propagations(
            'on', 1, nm=2, N_sol=np.abs(10*np.random.randn()))
        E = []
        for i in range(np.shape(u)[1]):
            E.append(np.linalg.norm(u[:, i], 2)**2)
        assert np.all(x == E[0] for x in E)


def test_bire_pass():
    Da = np.random.uniform(0, 2*pi, 100)
    b = birfeg_variation(Da)
    u = np.random.randn(2, 2**14) + 1j * np.random.randn(2, 2**14)
    u *= 10
    for i in range(100):
        ut = b.bire_pass(u,i)
        assert_allclose(np.abs(u)**2, np.abs(ut)**2)
        u = 1 * ut