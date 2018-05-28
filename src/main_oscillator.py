from __future__ import division, print_function
import numpy as np
import sys
from scipy.constants import c, pi
from joblib import Parallel, delayed
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from scipy.fftpack import fftshift, fft
import os
import time as timeit
os.system('export FONTCONFIG_PATH=/etc/fonts')
from functions import *
from time import time, sleep

@profile
def oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index,
             master_index, P0_p1, P0_s, f_p, f_s, p_pos, s_pos,
             splicers_vec, WDM_vec, M1, M2, Q_large, hf, Dop_large, dAdzmm, D_pic,
             pulse_pos_dict_or, plots, mode_names, ex, Dtheta, fopa):

    u = np.empty(
        [int_fwm.nm, len(sim_wind.t)], dtype='complex128')
    U = np.empty([int_fwm.nm,
                  len(sim_wind.t)], dtype='complex128')

    T0_p = TFWHM_p/2/(np.log(2))**0.5
    T0_s = TFWHM_s/2/(np.log(2))**0.5

    
    noise_new_or = noise_obj.noise_func_freq(int_fwm, sim_wind)
    u[:, :] = noise_obj.noise

    woff1 = (p_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0, :] += (P0_p1)**0.5 * np.exp(1j*(woff1)*sim_wind.t)

    woff2 = (s_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0, :] += (P0_s)**0.5 * np.exp(1j*(woff2) *
                                           sim_wind.t)

    U[:, :] = fftshift(fft(u[:, :]), axes = -1)

    sim_wind.w_tiled = np.tile(sim_wind.w + sim_wind.woffset, (int_fwm.nm, 1))
    master_index = str(master_index)

    ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                P0_s, f_p, f_s, 0, 0,  mode_names, master_index, '00', 'original pump', D_pic[0], plots)


    U_original_pump = np.copy(U[:, :])

    # Pass the original pump through the WDM1, port1 is in to the loop, port2
    # junk
    noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
    u[:, :], U[:, :] = WDM_vec[0].pass_through(
        (U[:, :], noise_new), sim_wind)[0]

    max_rounds = arguments_determine(-1)
    if fopa:
        print('Fibre amplifier!')
        max_rounds = 0
    ro = -1

    t_total = 0
    gam_no_aeff = -1j*int_fwm.n2*2*pi/sim_wind.lamda
    noise_new = noise_new_or*1

    while ro < max_rounds:

        ro += 1

        print('round', ro)
        pulse_pos_dict = [
            'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, 0, ro,  mode_names, master_index,
                    str(ro)+'1', pulse_pos_dict[3], D_pic[5], plots)

        for index_woble in range(len(int_fwm.Dz_vec)):
            Q , Dop = Q_large[index_woble], Dop_large[index_woble]
            int_fwm.woble_propagate(index_woble)                   
            u, U = pulse_propagation(u, U, int_fwm, M1, M2, Q,
                                     sim_wind, hf, Dop, dAdzmm,gam_no_aeff)
            u = Dtheta.bire_pass(u,index_woble)

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, -1, ro, mode_names, master_index,
                    str(ro)+'2', pulse_pos_dict[0], D_pic[2], plots)

        # pass through WDM2 port 2 continues and port 1 is out of the loop
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
        (out1, out2), (u[:, :], U[:, :]) = WDM_vec[1].pass_through(
            (U[:, :], noise_new), sim_wind)

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, -1, ro,  mode_names, master_index,
                    str(ro)+'3', pulse_pos_dict[1], D_pic[3], plots)

        # Splice7 after WDM2 for the signal
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)

        (u[:, :], U[:, :]) = splicers_vec[2].pass_through(
            (U[:, :], noise_new), sim_wind)[0]

        # Pass again through WDM1 with the signal now
        (u[:, :], U[:, :]) = WDM_vec[0].pass_through(
            (U_original_pump, U[:, :]), sim_wind)[0]

        ################################The outbound stuff#####################
        ex.exporter(index, int_fwm, sim_wind, out1, out2, P0_p1,
                    P0_s, f_p, f_s, -
                    1, ro,  mode_names, master_index, str(ro)+'4',
                    pulse_pos_dict[4], D_pic[6], plots)
    return None


def calc_P_out(U, U_original_pump, fv, t):
    U = np.abs(U)**2
    U_original_pump = np.abs(U_original_pump)**2
    freq_band = 2
    fp_id = np.where(U_original_pump == np.max(U_original_pump))[0][0]
    plom = fp_id+10
    fv_id = np.where(U[plom:] == np.max(U[plom:]))[0][0]
    fv_id += plom-1
    start, end = fv[fv_id] - freq_band, fv[fv_id] + freq_band
    i = np.where(
        np.abs(fv - start) == np.min(np.abs(fv - start)))[0][0]
    j = np.where(
        np.abs(fv - end) == np.min(np.abs(fv - end)))[0][0]
    E_out = simps(U[i:j]*(t[1] - t[0])**2, fv[i:j])
    P_out = E_out/(2*np.max(t))
    return P_out


@unpack_args
def formulate(index, n2, gama, alphadB, P_p, P_s, TFWHM_p, TFWHM_s, spl_losses,
              lamda_c, WDMS_pars, lamp, lams, num_cores, maxerr, ss, ram, plots,
              N, nt, nplot, master_index, nm, mode_names, pertb_vec, fopa, Deltaf):
    a_vec,dnerr,Dtheta, z_vec  = pertb_vec
    ex = Plotter_saver(plots, True)  # construct exporter
    "------------------propagation paramaters------------------"
    dz_less = 2
    Num_a = len(a_vec)
    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z_vec, nplot, dz_less,Num_a)
    lamda = lamp*1e-9  # central wavelength of the grid[m]
    fv_idler_int = 10  # safety for the idler to be spotted used only for idler power
    "-----------------------------f-----------------------------"
    Dtheta = birfeg_variation(Dtheta)
    "---------------------Grid&window-----------------------"
    print(lamp, lams,P_s)
    fv, where = fv_creator(lamp,lams,P_s, Deltaf, int_fwm)
    p_pos, s_pos = where
    print(p_pos, s_pos)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm, fv_idler_int)
    

    "----------------------------------------------------------"

    "---------------------Aeff-Qmatrixes-----------------------"
    M1, M2, betas, Q_large = fibre_parameter_loader(fv, a_vec, dnerr,
                                                    index, master_index,
                                                    filename='step_index_2m'
                                                    )

    "----------------------------------------------------------"

    "---------------------Loss-in-fibres-----------------------"
    slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0])/100
    loss = Loss(int_fwm, sim_wind, amax=None)

    
    int_fwm.alpha = loss.atten_func_full(fv)

    "----------------------------------------------------------"

    "--------------------Dispersion----------------------------"
    Dop_large = dispersion_operator(betas, int_fwm, sim_wind)
    "----------------------------------------------------------"

    "--------------------Noise---------------------------------"
    noise_obj = Noise(int_fwm, sim_wind)
    a = noise_obj.noise_func_freq(int_fwm, sim_wind)
    "----------------------------------------------------------"

    "---------------Formulate the functions to use-------------"
    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')

    keys = ['loading_data/green_dot_fopo/pngs/' +
            str(i)+str('.png') for i in range(7)]
    D_pic = [plt.imread(i) for i in keys]


    integrand = Integrand(ram, ss, cython = True, timing = False)
    dAdzmm = integrand.dAdzmm
    raman = raman_object(int_fwm.ram, int_fwm.how)
    raman.raman_load(sim_wind.t, sim_wind.dt, M2)
    hf = raman.hf
    "--------------------------------------------------------"

    "----------------------Formulate WDMS--------------------"
    if WDMS_pars == 'signal_locked':

        Omega = 2*pi*c/(lamp*1e-9) - 2*pi*c/(lams*1e-9)
        omegai = 2*pi*c/(lamp*1e-9) + Omega
        lami = 1e9*2*pi*c/(omegai)
        WDMS_pars = ([lamp, lams],  # WDM up downs in wavelengths [m]
                     [lami, lams],
                     [lami, lamp],
                     [lami, lams])

    WDM_vec = [WDM(i[0], i[1], sim_wind.fv, c, fopa,nm)
               for i in WDMS_pars]  # WDM up downs in wavelengths [m]

    "--------------------------------------------------------"

    "----------------------Formulate splicers--------------------"
    splicers_vec = [Splicer(fopa = fopa,loss=i) for i in spl_losses]
    "------------------------------------------------------------"

    f_p, f_s = 1e-3*c/lamp, 1e-3*c/lams

    oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index, master_index, P_p, P_s, f_p, f_s, p_pos, s_pos, splicers_vec,
             WDM_vec, M1, M2, Q_large, hf, Dop_large, dAdzmm, D_pic, pulse_pos_dict_or, plots, mode_names, ex, Dtheta, fopa)
    return None


def main():
    "-----------------------------Stable parameters----------------------------"
    # Number of computing cores for sweep
    num_cores = arguments_determine(1)
    # maximum tolerable error per step in integration
    maxerr = 1e-13
    ss = 1                                  # includes self steepening term
    ram = 'on'                              # Raman contribution 'on' if yes and 'off' if no
    if arguments_determine(-1) == 0:
        fopa = True                         # If FOPA true or if FOPO then false
    else:
        fopa = False
    plots = True                            # Do you want plots, be carefull it makes the code very slow!
    N = 12                                  # 2**N grid points
    nt = 2**N                               # number of grid points
    nplot = 2                               # number of plots within fibre min is 2
    # Number of modes (include degenerate polarisation)
    
    nm = 2
    mode_names = ['LP01a', 'LP01b']         # Names of modes for plotting
    if 'mpi' in sys.argv:
        method = 'mpi'
    elif 'joblib' in sys.argv:
        method = 'joblib'
    else:
        method = 'single'
    "--------------------------------------------------------------------------"
    stable_dic = {'num_cores': num_cores, 'maxerr': maxerr, 'ss': ss, 'ram': ram, 'plots': plots,
                  'N': N, 'nt': nt, 'nplot': nplot, 'nm': nm, 'mode_names': mode_names, 'fopa':fopa}
    "------------------------Can be variable parameters------------------------"
    n2 = 2.5e-20                            # Nonlinear index [m/W]
    gama = 10e-3                            # Overwirtes n2 and Aeff w/m        
    alphadB = np.array([0,0])              # loss within fibre[dB/m]
    z = 1000                                 # Length of the fibre
    P_p = [10]
    P_s = 100e-3
    TFWHM_p = 0                             # full with half max of pump
    TFWHM_s = 0                             # full with half max of signal
    spl_losses = [[0, 0, 1.], [0, 0, 1.2], [0, 0, 1.3], [
        0, 0, 1.4]]                 # loss of each type of splices [dB]
    spl_losses = [0, 0, 1.4]
    loada_vec = True
    a_med = 2.19e-6
    a_err = 0.008 #[125um +- 1um ]
    dnerr_med = 0.0002
    cutting = 100
    Num_a = 1000
    if loada_vec and os.path.isfile('loading_data/a_vec.hdf5'):   
        D = read_variables('a_vec', '0', filepath='loading_data/')
        a_vec = D['a_vec']
        dnerr = D['dnerr']
        Dtheta = D['Dtheta']
    else:
        a_vec = np.random.normal(a_med,a_err * a_med, Num_a )
        dnerr = np.random.normal(0, dnerr_med, len(a_vec))
        Dtheta = np.random.normal(0, pi, len(a_vec))
        D = {'a_vec':a_vec, 'dnerr':dnerr, 'Dtheta':Dtheta}
        os.system('rm loading_data/a_vec.hdf5')
        save_variables('a_vec', '0', filepath='loading_data/', **D)

    z_vec = np.linspace(0, z, len(a_vec)+1)
    pertb_vec = [[a_vec,dnerr,Dtheta, z_vec]] # pertubation vector for dn and a_vec
    

    if cutting < len(a_vec):
        pertb_vec += [[j[:-(cutting*i)] for j in (a_vec, dnerr, Dtheta, z_vec)]\
                       for i in range(1,int(len(a_vec)/cutting))]



    lamda_c = 1051.85e-9
    WDMS_pars = ([1050., 1392.85],
                 [1747.585,  1392.85])  # WDM up downs in wavelengths [m]

    Deltaf = 32
    
    lamp = 1550
    lams = 1392.85
    var_dic = {'n2': n2, 'gama': gama, 'alphadB': alphadB, 'P_p': P_p,
               'P_s': P_s, 'TFWHM_p': TFWHM_p, 'TFWHM_s': TFWHM_s,
               'spl_losses': spl_losses,
               'lamda_c': lamda_c, 'WDMS_pars': WDMS_pars,
               'lamp': lamp, 'lams': lams,
               'pertb_vec':pertb_vec, 'Deltaf':Deltaf}

    "--------------------------------------------------------------------------"
    outside_var_key = 'P_p'
    inside_var_key = 'pertb_vec'
    inside_var = var_dic[inside_var_key]
    outside_var = var_dic[outside_var_key]
    del var_dic[outside_var_key]
    del var_dic[inside_var_key]
    "----------------------------Simulation------------------------------------"
    D_ins = [{'index': i, inside_var_key: insvar}
             for i, insvar in enumerate(inside_var)]

    large_dic = {**stable_dic, **var_dic}

    if len(inside_var) < num_cores:
        num_cores = len(inside_var)

    profiler_bool = arguments_determine(0)
    for kk, variable in enumerate(outside_var):
        create_file_structure(kk)

        _temps = create_destroy(inside_var, str(kk))
        _temps.prepare_folder()
        large_dic['master_index'] = kk
        large_dic[outside_var_key] = variable
        if profiler_bool:
            for i in range(len(D_ins)):
                formulate(**{**D_ins[i], ** large_dic})
        elif method == 'mpi':
            iterables = ({**D_ins[i], ** large_dic} for i in range(len(D_ins)))
            with MPIPoolExecutor() as executor:
                A = executor.map(formulate, iterables)
        else:
            A = Parallel(n_jobs=num_cores)(delayed(formulate)(**{**D_ins[i], ** large_dic}) for i in range(len(D_ins)))
        _temps.cleanup_folder()
    consolidate_hdf5_steps(len(outside_var), len(
        inside_var), filepath='loading_data/step_data/')
    print('\a')
    return None

if __name__ == '__main__':
    start = time()
    main()
    dt = time() - start
    print(dt, 'sec', dt/60, 'min', dt/60/60, 'hours')
