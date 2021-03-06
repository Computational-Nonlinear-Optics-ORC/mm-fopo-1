import numpy as np
import sys
sys.path.append('src')
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
from overlaps import fibre_overlaps_loader



@profile
def oscilate(sim_wind, int_fwm, noise_obj, index,
             master_index,
             splicers_vec, WDM_vec, M1, M2, Q_large, hf, Dop, dAdzmm, D_pic,
             pulse_pos_dict_or, plots, mode_names, ex, fopa, D_param, pm):

    u = np.empty(
        [int_fwm.nm, len(sim_wind.t)], dtype='complex128')
    U = np.empty([int_fwm.nm,
                  len(sim_wind.t)], dtype='complex128')
    p1_pos = D_param['where'][1]
    p2_pos = D_param['where'][4]
    s_pos = D_param['where'][2]


    noise_new_or = noise_obj.noise_func_freq(int_fwm, sim_wind)
    u[:, :] = noise_obj.noise

    woff1 = (p1_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0, :] += (D_param['P_p1'])**0.5 * np.exp(1j*(woff1)*sim_wind.t)



    woff2 = (s_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[0, :] += (D_param['P_s'])**0.5 * np.exp(1j*(woff2) *
                                           sim_wind.t)


    woff3 = (p2_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[1, :] += (D_param['P_p2'])**0.5 * np.exp(1j*(woff3) *
                                           sim_wind.t)

    U[:, :] = fftshift(fft(u[:, :]), axes = -1)
    w_tiled = np.tile(sim_wind.w + sim_wind.woffset, (int_fwm.nm, 1))
    master_index = str(master_index)

    ex.exporter(index, int_fwm, sim_wind, u, U, D_param,
                 0, 0,  mode_names, master_index,
                 '00', 'original pump', D_pic[0], plots)


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
    dz,dzstep,maxerr = int_fwm.dz,int_fwm.z,int_fwm.maxerr

    Dop = np.ascontiguousarray(Dop / 2)
    w_tiled = np.ascontiguousarray(w_tiled)
    tsh = sim_wind.tsh

    while ro < max_rounds:

        ro += 1

        print('round', ro)
        pulse_pos_dict = [
            'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]

        ex.exporter(index, int_fwm, sim_wind, u, U, D_param, 0, ro,  mode_names, master_index,
                    str(ro)+'1', pulse_pos_dict[3], D_pic[5], plots)


        U, dz = pulse_propagation(u, dz, dzstep, maxerr, M1, M2, Q_large, w_tiled, tsh, hf, Dop, gam_no_aeff)


        ex.exporter(index, int_fwm, sim_wind, u, U, D_param, -1, ro, mode_names, master_index,
                    str(ro)+'2', pulse_pos_dict[0], D_pic[2], plots)
        
        # pass through WDM2 port 2 continues and port 1 is out of the loop
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
        (u[:, :], U[:, :]),(out1, out2) = WDM_vec[1].pass_through(
            (U[:, :], noise_new), sim_wind)

        ex.exporter(index, int_fwm, sim_wind, u, U, D_param, -1, ro,  mode_names, master_index,
                    str(ro)+'3', pulse_pos_dict[1], D_pic[3], plots)

        # Splice7 after WDM2 for the signal
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)

        (u[:, :], U[:, :]) = splicers_vec[2].pass_through(
            (U[:, :], noise_new), sim_wind)[0]
        # Modulate the phase to be in phase with what is coming in
        pm.modulate(U_original_pump, U)
        
        # Pass again through WDM1 with the signal now
        (u[:, :], U[:, :]) = WDM_vec[0].pass_through(
            (U_original_pump, U[:, :]), sim_wind)[0]

        ################################The outbound stuff#####################
        ex.exporter(index, int_fwm, sim_wind, out1, out2, D_param, -
                    1, ro,  mode_names, master_index, str(ro)+'4',
                    pulse_pos_dict[4], D_pic[6], plots)
    if max_rounds == 0:
        max_rounds =1
    consolidate(max_rounds, int_fwm,master_index, index)
    return None



@unpack_args
def formulate(index, n2, alphadB, P_p1, P_p2, P_s, spl_losses,
              lamda_c, WDMS_pars, lamp1, lamp2, lams, num_cores,
              maxerr, ss, ram, plots,
              N, nt, master_index, nm, mode_names, fopa,z):
    ex = Plotter_saver(plots, True)  # construct exporter
    "------------------propagation paramaters------------------"
    dz_less = 2
    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z, dz_less)
    lamda = lamp1*1e-9  # central wavelength of the grid[m]
    "---------------------Grid&window-----------------------"
    fv, D_freq = fv_creator(lamp1,lamp2, lams, int_fwm)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm)
    "----------------------------------------------------------"

    "---------------------Aeff-Qmatrixes-----------------------"
    M1, M2, Q_large = fibre_overlaps_loader(sim_wind.dt)
    betas = load_disp_paramters(sim_wind.w0)


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
    print(WDMS_pars)
    "----------------------Formulate WDMS--------------------"
    if WDMS_pars[-1] == 'WDM':
        WDM_vec = [WDM(i[0], i[1], sim_wind.fv, fopa,with_resp)
                   for i, with_resp in zip(WDMS_pars[:-1], ('LP01', 'LP11'))]  # WDM up downs in wavelengths [m]
    elif WDMS_pars[-1] == 'prc':
        WDM_vec = [Perc_WDM(D_freq['where'], i, sim_wind.fv, fopa)
                   for i in WDMS_pars[:-1]]  # WDM up downs in wavelengths [m]
    elif WDMS_pars[-1] == 'bandpass':
        WDM_vec = [Bandpass_WDM(D_freq['where'], i, sim_wind.fv, fopa)
                   for i in WDMS_pars[:-1]]  # WDM up downs in wavelengths [m]

    "--------------------------------------------------------"

    "----------------------Formulate splicers--------------------"
    splicers_vec = [Splicer(fopa = fopa, loss=i) for i in spl_losses]
    "------------------------------------------------------------"

    D_param = {**D_freq, **{'P_p1': P_p1, 'P_p2': P_p2, 'P_s': P_s}} 

    pm = Phase_modulation_infase_WDM(D_freq['where'], WDM_vec[0])

    oscilate(sim_wind, int_fwm, noise_obj, index, master_index, splicers_vec,
             WDM_vec, M1, M2, Q_large, hf, Dop_large, dAdzmm, D_pic,
             pulse_pos_dict_or, plots, mode_names, ex, fopa,D_param,pm )
    return None


def main():
    "-----------------------------Stable parameters----------------------------"
    # Number of computing cores for sweep
    num_cores = arguments_determine(1)
    # maximum tolerable error per step in integration
    maxerr = 1e-14
    ss = 1                                  # includes self steepening term
    ram = 'on'                              # Raman contribution 'on' if yes and 'off' if no
    if arguments_determine(-1) == 0:
        fopa = True                         # If no oscillations then the WDMs are deleted to 
                                            # make the system in to a FOPA
    else:
        fopa = False
    plots = True                           # Do you want plots, be carefull it makes the code very slow!
    N = 12                                   # 2**N grid points
    nt = 2**N                               # number of grid points
    nplot = 2                               # number of plots within fibre min is 2

    
    nm = 2
    mode_names = ['LP01', 'LP11a']         # Names of modes for plotting
    if 'mpi' in sys.argv:
        method = 'mpi'
    elif 'joblib' in sys.argv:
        method = 'joblib'
    else:
        method = 'single'
    "--------------------------------------------------------------------------"
    stable_dic = {'num_cores': num_cores, 'maxerr': maxerr, 'ss': ss, 'ram': ram, 'plots': plots,
                  'N': N, 'nt': nt, 'nm': nm, 'mode_names': mode_names, 'fopa':fopa}
    "------------------------Can be variable parameters------------------------"
    n2 = 2.5e-20                            # Nonlinear index [m/W]
    alphadB = np.array([0,0])              # loss within fibre[dB/m]
    z = [5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]  # Length of the fibre
    z = np.linspace(1, 1000, 100)
    P_p1 = dbm2w(30.5 - 3)
    P_p2 = dbm2w(30.5 - 4)
    P_s = dbm2w(30.5 - 3 - 24)#1e-3#1e-3
    spl_losses = [0, 0, 1.]
    lamda_c = 1.5508e-6
    WDMS_pars = ([1549., 1550.],
                 [1555,  1556.], 'WDM')  # WDM up downs in wavelengths [m]
    
    WDMS_pars = ([100, 100, 20, 0, 100, 0],
                 [100, 100, 100, 0, 100, 0], 'bandpass')  # WDM up downs in wavelengths [m]
    # Bandpass filter system [m]
    #WDMS_pars = []
    #for i in range(5, 100, 5):
    #    WDMS_pars.append([[100, 100, i, 0, 100, 0],
    #             [100, 100, 100, 0, 0, 0], 'bandpass'])

    lamp1 = 1549
    lamp2 = [1553.50048583,1554.50204646]#, 1555]
    lamp2 = 1553.50048583
    #lamp2 = [1553.25]
    lams = np.linspace(1549, 1549+5, 256)
    #lams = [1550]
    lams = lams[5:]
    var_dic = {'n2': n2, 'alphadB': alphadB,
               'P_p1': P_p1, 'P_p2': P_p2, 'P_s': P_s,
               'spl_losses': spl_losses,
               'lamda_c': lamda_c, 'WDMS_pars': WDMS_pars,
               'lamp1': lamp1,'lamp2': lamp2, 'lams': lams, 'z':z}

    "--------------------------------------------------------------------------"
    outside_var_key = 'z'
    inside_var_key = 'lams'
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
    print('\a')
    return None

if __name__ == '__main__':
    start = time()
    main()
    dt = time() - start
    print(dt, 'sec', dt/60, 'min', dt/60/60, 'hours')
