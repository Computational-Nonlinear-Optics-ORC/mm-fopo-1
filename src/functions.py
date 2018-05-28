# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import os
from cython_files.cython_integrand import *
import numpy as np
from scipy.constants import pi, c
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from math import factorial
from integrand_and_rk import *
from data_plotters_animators import *
from step_index import fibre_creator
from step_index_functions import save_variables_step
import cmath
from time import time
from scipy.fftpack import fft, ifft
from scipy.fftpack import ifftshift
phasor = np.vectorize(cmath.polar)
from functools import wraps
from step_index import Sidebands
# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func


def arguments_determine(j):
    """
    Makes sence of the arguments that are passed through from sys.agrv. 
    Is used to fix the mpi4py extra that is given. Takes in the possition 
    FROM THE END of the sys.argv inputs that you require (-1 would be the rounds
    for the oscillator).
    """
    A = []
    a = np.copy(sys.argv)
    for i in a[::-1]:
        try:
            A.append(int(i))
        except ValueError:
            continue
    return A[j]


def unpack_args(func):
    if 'mpi' in sys.argv:
        @wraps(func)
        def wrapper(args):
            return func(**args)

        return wrapper
    else:
        return func


def my_arange(a, b, dr, decimals=6):
    res = [a]
    k = 1
    while res[-1] < b:
        tmp = round(a + k*dr, decimals)
        if tmp > b:
            break
        res.append(tmp)
        k += 1

    return np.asarray(res)


def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
               dBm(float): power in units of dBm
       Returns::
               Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


def w2dbm(W, floor=-100):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
               W(float): power in units of W
       Returns::
               Power in units of dBm(float)
    """
    if type(W) != np.ndarray:
        if W > 0:
            return 10. * np.log10(W) + 30
        elif W == 0:
            return floor
        else:
            print(W)
            raise(ZeroDivisionError)
    a = 10. * (np.ma.log10(W)).filled(floor/10-3) + 30
    return a


class raman_object(object):

    def __init__(self, a, b=None):
        self.on = a
        self.how = b
        self.hf = None

    def raman_load(self, t, dt, M2):
        if self.on == 'on':
            if self.how == 'analytic':
                print(self.how)
                t11 = 12.2e-3     # [ps]
                t2 = 32e-3       # [ps]
                # analytical response
                htan = (t11**2 + t2**2)/(t11*t2**2) * \
                    np.exp(-t/t2*(t >= 0))*np.sin(t/t11)*(t >= 0)
                # Fourier transform of the analytic nonlinear response
                self.hf = fft(htan)
            elif self.how == 'load':
                # loads the measured response (Stolen et al. JOSAB 1989)
                mat = loadmat('loading_data/silicaRaman.mat')
                ht = mat['ht']
                t1 = mat['t1']
                htmeas_f = InterpolatedUnivariateSpline(t1*1e-3, ht)
                htmeas = htmeas_f(t)
                htmeas *= (t > 0)*(t < 1)  # only measured between +/- 1 ps)
                htmeas /= (dt*np.sum(htmeas))  # normalised
                # Fourier transform of the measured nonlinear response
                self.hf = fft(htmeas)
                self.hf = np.tile(self.hf, (len(M2[1, :]), 1))
            else:
                self.hf = None

            return self.hf


def dispersion_operator(betas, int_fwm, sim_wind):
    """
    Calculates the dispersion operator in rad/m units
    Inputed are the dispersion operators at the omega0
    Local include the taylor expansion to get these opeators at omegac 
    Returns Dispersion operator
    """

    w = sim_wind.w + sim_wind.woffset
    betap = np.tile(betas, (int_fwm.nm, 1, 1))

    Dop = np.zeros((betas.shape[0], int_fwm.nm, w.shape[0]), dtype=np.complex)

    for i in range(Dop.shape[0]):
        Dop[i, :, :] -= fftshift(int_fwm.alpha/2, axes=-1)

    for i in range(int_fwm.nm-1, -1, -1):
        betap[i, :, 0] = betap[i, :, 0] - betap[0, :, 0]
        betap[i, :, 1] = betap[i, :, 1] - betap[0, :, 1]

    for k, b in enumerate(betap):
        for l, bb in enumerate(b):
            for m, bbb in enumerate(bb):
                Dop[l, k, :] = Dop[l, k, :]-1j*(w**m * bbb / factorial(m))
    return Dop


def load_step_index_params(filename, filepath):
    with h5py.File(filepath+filename+'.hdf5', 'r') as f:
        D = {}
        for i in f.keys():
            try:
                D[str(i)] = f.get(str(i)).value
            except AttributeError:
                pass
    a_vec, fv, dnerr,  M1, M2, betas, Q_large, dnerr =\
        D['a_vec'], D['fv'], D['dnerr'], D['M1'], \
        D['M2'], D['betas'], D['Q_large'], D['dnerr']

    return a_vec, fv, M1, M2, betas, Q_large, dnerr, D


def consolidate_hdf5_steps(master_index_l, size_ins, filepath):
    """
    Puts all exported HDF5 files created to one and saves it for future 
    computational saving time. 
    """
    if os.path.isfile(filepath+'step_index_2m.hdf5'):
        os.system('rm ' + filepath+'step_index_2m.hdf5')
    for master_index in range(master_index_l):
        for index in range(size_ins):
            layer = str(int(size_ins * master_index + index))
            filename = 'step_index_2m'+'_new_'+str(master_index)+'_'+str(index)
            D = load_step_index_params(filename, filepath)[-1]
            save_variables('step_index_2m', layer, filepath=filepath, **D)
            os.system('rm '+filepath+filename+'.hdf5')
    return None


def find_large_block_data_full(already_done, layer_old, filepath, filename, D_now):
    """
    Searches the large block file to see if the data is already cached
    Returns a bool. 
    """
    with h5py.File(filepath+filename+'.hdf5', 'r') as f:
        for layer_old in f.keys():
            D = [f.get(layer_old + '/' + str(i)
                       ).value for i in ('a_vec', 'fv', 'dnerr')]
            try:
                already_done = np.array(
                    [np.allclose(D_now[i], D[i]) for i in range(3)]).all()
            except ValueError:
                pass
            if already_done:
                print('found in large')
                break
    return already_done, layer_old


def find_small_block_data_full(already_done, layer_old, filepath, filename, D_now):
    """
    Searches the small block files to see if the data is already cached
    Returns a bool. 
    """

    files = os.listdir(filepath)
    if 'step_index_2m.hdf5' in files:
        files.remove('step_index_2m.hdf5')
    f = None
    for file in files:
        with h5py.File(filepath+file, 'r') as f:
            D = [f.get(str(i)).value for i in ('a_vec', 'fv', 'dnerr')]
            try:
                already_done = np.array(
                    [np.allclose(D_now[i], D[i]) for i in range(3)]).all()
            except ValueError:
                pass
        if already_done:
            print('found in small')
            f = file
            break
    return already_done, f


def fibre_parameter_loader(fv, a_vec, dnerr, index, master_index,
                           filename, filepath='loading_data/step_data/'):
    """
    This function tried to save time in computation of the step index dipsersion. It
    compares the hdf5 file that was exported from a previous computation if the inputs dont
    fit then it calls the eigenvalue solvers. It has also been extended to look if previous
    results within the same computation hold the same results. Tested on parallell.
    """

    index = str(index)
    master_index = str(master_index)

    ############################Total step index computation################################
    if os.listdir(filepath) == []:  # No files in dir, all to be calc
        print('No files in dir, calculating new radius:', filepath)
        Export_dict = fibre_creator(a_vec, fv, dnerr, filepath=filepath)[-1]
        save_variables_step(filename+'_new_'+master_index+'_'+index,
                            filepath=filepath, **Export_dict)
        M1, M2, betas, Q_large = Export_dict['M1'], Export_dict['M2'], \
            np.asanyarray(Export_dict['betas']), Export_dict['Q_large']
        return M1, M2, betas, Q_large
    #########################################################################################

    ####################Try and find entire blocks in large or small files###################
    D_now = [a_vec, fv, dnerr]
    already_done = False
    layer_old = False

    try:
        # Try and find in the consolidated
        already_done, layer_old =\
            find_large_block_data_full(
                already_done, layer_old, filepath, filename, D_now)
    except OSError:
        pass

    if not(already_done):
        # Try and find in the normal ones
        try:
            already_done, file = \
                find_small_block_data_full(
                    already_done, layer_old, filepath, filename, D_now)
        except OSError:
            pass
    if already_done:
        # If the entire computation is already done then simply load and save variables
        if layer_old:
            D = read_variables(filename, layer_old, filepath=filepath)
        else:
            D = load_step_index_params(file[:-5], filepath)[-1]

        if os.path.isfile(filepath+filename+'_new_' +
                          master_index+'_'+index+'.hdf5'):
            os.system('rm ' + filepath+filename+'_new_' +
                      master_index+'_'+index+'.hdf5')

        save_variables_step(filename+'_new_'+master_index +
                            '_'+index,  filepath=filepath, **D)

        M1, M2, betas, Q_large = D['M1'], D['M2'], D['betas'], D['Q_large']
        return M1, M2, betas, Q_large
    ##########################################################################################
    M1, M2, betas, Q_large = find_small_block_data_small(
        D_now, filename, filepath, master_index, index)
    return M1, M2, betas, Q_large


def compare_single_data(fv_old, fv, a_vec, dnerr, a_vec_old,
                        dnerr_old, betas, betas_old, Q_large,
                        Q_large_old, found):
    try:
        if np.allclose(fv_old, fv):
            for j in range(len(a_vec_old)):
                i_vec = np.where(np.isclose(a_vec, [a_vec_old[j]]) *
                                 np.isclose(dnerr, [dnerr_old[j]]))[0]
                for i in i_vec:
                    print('found', a_vec_old[j])
                    Q_large[i] = Q_large_old[j, :, :]
                    betas[i] = betas_old[j, :]
                    found[i] = True
    except ValueError:
        pass
    return found, Q_large, betas


def find_small_block_data_small(D_now, filename, filepath, master_index, index):
    """
    Searches the block files to see if there is any
    data that has already been calculated prior to launch, calculates what
    is left. First it tries in the consolidated file and then
    in the small blocks(the later helps in parallel) Only works for 2 modes!
    """
    a_vec, fv, dnerr = D_now
    files = os.listdir(filepath)
    if 'step_index_2m.hdf5' in files:
        files.remove('step_index_2m.hdf5')
    Q_large = [0 for i in a_vec]
    betas = [0 for i in a_vec]
    found = np.zeros(len(a_vec))

    ####################looking in the large block cashes###############
    try:
        with h5py.File(filepath+filename+'.hdf5', 'r') as f:
            for layer_old in f.keys():
                D = read_variables(filename, layer_old, filepath)
                a_vec_old, fv_old, M1_old, M2_old, \
                    betas_old, Q_large_old, dnerr_old = \
                    D['a_vec'], D['fv'], D['M1'], D['M2'],\
                    D['betas'], D['Q_large'], D['dnerr']

                found, Q_large, betas = \
                    compare_single_data(fv_old, fv, a_vec, dnerr, a_vec_old,
                                        dnerr_old, betas, betas_old, Q_large,
                                        Q_large_old, found)

                if (found == True).all():
                    break
    except OSError:
        pass
    ######################################################################

    ####################looking in the small block cashes###############
    try:
        for file in files:
            a_vec_old, fv_old, M1_old, M2_old, betas_old,\
                Q_large_old, dnerr_old = \
                load_step_index_params(file[:-5], filepath)[:-1]
            found, Q_large, betas = \
                compare_single_data(fv_old, fv, a_vec, dnerr, a_vec_old,
                                    dnerr_old, betas, betas_old, Q_large,
                                    Q_large_old, found)

            if (found == True).all():
                break
    except OSError:
        pass
    ######################################################################

    found = np.array([np.nan if i else 1 for i in found])
    # What is missing? Fix the array and send it though to calculate them.
    # dnerr_temp = np.ones_like(dnerr) # Dirty fix because dnerr can be zero
    a_vec_needed, dnerr_needed = a_vec * found, dnerr * found
    a_vec_needed, dnerr_needed = a_vec_needed[np.isfinite(a_vec_needed)], \
        dnerr_needed[np.isfinite(dnerr_needed)]

    if a_vec_needed.any():
        print('Doing some extra calculations for data not cached')
        print(a_vec_needed, dnerr_needed)
        Export_dict = fibre_creator(
            a_vec_needed, fv, dnerr_needed, filepath=filepath)[-1]
        M1, M2, betas_new, Q_large_new = Export_dict['M1'], Export_dict['M2'], \
            np.asanyarray(Export_dict['betas']), Export_dict['Q_large']
        count = 0
        for i in range(len(a_vec)):
            if np.isfinite(found[i]):

                Q_large[i] = Q_large_new[count, :, :]
                betas[i] = betas_new[count, :]
                count += 1
    else:
        M1, M2 = M1_old, M2_old
    Q_large = np.asanyarray(Q_large)
    betas = np.asanyarray(betas)

    Export_dict = {'M1': M1, 'M2': M2,
                   'Q_large': Q_large, 'betas': betas,
                   'a_vec': a_vec, 'fv': fv, 'dnerr': dnerr}
    save_variables_step(filename+'_new_'+master_index+'_'+index,
                        filepath=filepath, **Export_dict)
    return M1, M2, betas, Q_large


class sim_parameters(object):

    def __init__(self, n2, nm, alphadB):
        self.n2 = n2
        self.nm = nm
        self.alphadB = alphadB
        try:
            temp = len(self.alphadB)
        except TypeError:
            self.alphadB = np.array([self.alphadB])
        if self.nm > len(self.alphadB):
            print('Asserting same loss per mode')
            self.alphadB = np.empty(nm)
            self.alphadB = np.tile(alphadB, (nm))
        elif self.nm < len(self.alphadB):
            print('To many losses for modes, apending!')
            for i in range(nm):
                self.alphadB[i] = alphadB[i]
        else:
            self.alphadB = alphadB

    def general_options(self, maxerr, raman_object,
                        ss='1', ram='on', how='load'):
        self.maxerr = maxerr
        self.ss = ss
        self.ram = ram
        self.how = how
        return None

    def propagation_parameters(self, N, z_vec, nplot, dz_less, Num_a):
        self.N = N
        self.nt = 2**self.N
        self.nplot = nplot
        self.tot_z = np.max(z_vec)
        self.z_vec = z_vec
        self.Dz_vec = np.array([self.z_vec[i + 1] - self.z_vec[i]
                                for i in range(len(self.z_vec)-1)])
        self.dzstep_vec = self.Dz_vec
        self.dz = np.min([self.dzstep_vec[0]/2, 1])
        return None

    def woble_propagate(self, i):
        self.dzstep = self.dzstep_vec[i]
        return None


class sim_window(object):

    def __init__(self, fv, lamda, lamda_c, int_fwm, fv_idler_int):
        self.fv = fv
        self.lamda = lamda
        self.fmed = 0.5*(fv[-1] + fv[0])*1e12  # [Hz]
        self.deltaf = np.max(self.fv) - np.min(self.fv)  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1/self.df  # Time window (period)[ps]
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12  # [rad/ps]
        self.w0 = 2*pi*self.fmed  # central angular frequency [rad/s]
        self.tsh = 1/self.w0*1e12  # shock time [ps]
        self.dt = self.T/int_fwm.nt  # timestep (dt)     [ps]
        self.t = (range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2)*self.dt
        # angular frequency vector [rad/ps]
        self.w = 2*pi * np.append(
            range(0, int(int_fwm.nt/2)),
            range(int(-int_fwm.nt/2), 0, 1))/self.T

        self.lv = 1e-3*c/self.fv

        self.fv_idler_int = fv_idler_int
        self.fv_idler_tuple = (
            self.fmed*1e-12 - fv_idler_int, self.fmed*1e-12 + fv_idler_int)


class Loss(object):

    def __init__(self, int_fwm, sim_wind, amax=None, apart_div=8):
        """
        Initialise the calss Loss, takes in the general parameters and 
        the freequenbcy window. From that it determines where the loss will become
        freequency dependent. With the default value being an 8th of the difference
        of max and min.
        Note: From w-fopo onwards we introduce loss per mode which means we look at
        a higher dim array. 

        """
        self.alpha = int_fwm.alphadB/4.343
        if amax is None:
            self.amax = self.alpha
        else:
            self.amax = amax/4.343

        self.flims_large = (np.min(sim_wind.fv), np.max(sim_wind.fv))
        try:
            temp = len(apart_div)
            self.begin = apart_div[0]
            self.end = apart_div[1]
        except TypeError:

            self.apart = np.abs(self.flims_large[1] - self.flims_large[0])
            self.apart /= apart_div
            self.begin = self.flims_large[0] + self.apart
            self.end = self.flims_large[1] - self.apart

    def atten_func_full(self, fv):
        aten = np.zeros([len(self.alpha), len(fv)])

        a_s = ((self.amax - self.alpha) / (self.flims_large[0] - self.begin),

               (self.amax - self.alpha) / (self.flims_large[1] - self.end))
        b_s = (-a_s[0] * self.begin, -a_s[1] * self.end)

        for i, f in enumerate(fv):
            if f <= self.begin:
                aten[:, i] = a_s[0][:] * f + b_s[0][:]
            elif f >= self.end:
                aten[:, i] = a_s[1][:] * f + b_s[1][:]
            else:
                aten[:, i] = 0
        for i in range(len(self.alpha)):
            aten[i, :] += self.alpha[i]
        return aten

    def plot(self, fv):
        fig = plt.figure()
        y = self.atten_func_full(fv)
        for l, i in enumerate(y):
            plt.plot(fv, i, label='mode '+str(l))
        plt.xlabel("Frequency (Thz)")
        plt.ylabel("Attenuation (cm -1 )")
        plt.legend()
        plt.savefig(
            "loss_function_fibre.png", bbox_inches='tight')
        plt.close(fig)


class WDM(object):

    def __init__(self, x1, x2, fv, c, fopa=False, nm=1):
        """
        This class represents a 2x2 WDM coupler. The minimum and maximums are
        given and then the object represents the class with WDM_pass the calculation
        done.
        """
        self.l1 = x1   # High part of port 1
        self.l2 = x2  # Low wavelength of port 1
        self.f1 = 1e-3 * c / self.l1   # High part of port 1
        self.f2 = 1e-3 * c / self.l2  # Low wavelength of port 1
        self.omega = 0.5*pi/np.abs(self.f1 - self.f2)
        self.phi = 2*pi - self.omega*self.f2
        self.fv = fv
        self.fv_wdm = self.omega*fv+self.phi

        nt = len(self.fv)
        shape = (nm, nt)
        eps = np.sin(self.fv_wdm)
        eps2 = 1j*np.cos(self.fv_wdm)
        eps = np.tile(eps, (nm, 1))
        eps2 = np.tile(eps2, (nm, 1))
        self.A = np.array([[eps, eps2],
                           [eps2, eps]])
        if fopa:
            self.U_calc = self.U_calc_over
        return None

    def U_calc_over(self, U_in):
        return U_in

    def U_calc(self, U_in):
        """
        Uses the array defined in __init__ to calculate 
        the outputed amplitude in arbitary units

        """

        Uout = (self.A[0, 0] * U_in[0] + self.A[0, 1] * U_in[1],)
        Uout += (self.A[1, 0] * U_in[0] + self.A[1, 1] * U_in[1],)

        return Uout

    def pass_through(self, U_in, sim_wind):
        """
        Passes the amplitudes through the object. returns the u, U and Uabs
        in a form of a tuple of (port1,port2)
        """

        U_out = self.U_calc(U_in)
        u_out = ()
        for i, UU in enumerate(U_out):
            u_out += (ifft(fftshift(UU, axes=-1)),)
        return ((u_out[0], U_out[0]), (u_out[1], U_out[1]))

    def il_port1(self, fv_sp=None):
        """
        For visualisation of the wdm loss of port 1.
        If no input is given then it is plottedin the freequency 
        vector that the function is defined by. You can however 
        give an input in wavelength.
        """
        if fv_sp is None:
            return (np.sin(self.omega*self.fv+self.phi))**2
        else:
            return (np.sin(self.omega*(1e-3*c/fv_sp)+self.phi))**2

    def il_port2(self, fv_sp=None):
        """
        Like il_port1 but with cosine (oposite)
        """
        if fv_sp is None:
            return (np.cos(self.omega*self.fv+self.phi))**2
        else:
            return (np.cos(self.omega*(1e-3*c/fv_sp) + self.phi))**2

    def plot(self, filename=False, xlim=False):
        fig = plt.figure()
        plt.plot(1e-3*c/self.fv, self.il_port1(), label="%0.2f" %
                 (self.l1) + ' nm port')
        plt.plot(1e-3*c/self.fv, self.il_port2(), label="%0.1f" %
                 (self.l2) + ' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        plt.xlabel(r'$\lambda (n m)$')
        plt.ylabel('Power Ratio')
        if xlim:
            plt.xlim(xlim)
        if filename:
            plt.savefig(filename+'.png')
        else:
            plt.show()
        plt.close(fig)
        return None

    def plot_dB(self, lamda, filename=False):
        fig = plt.figure()
        plt.plot(lamda, 10*np.log10(self.il_port1(lamda)),
                 label="%0.2f" % (self.l1*1e9) + ' nm port')
        plt.plot(lamda, 10*np.log10(self.il_port2(lamda)),
                 label="%0.2f" % (self.l2*1e9) + ' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        plt.xlabel(r'$\lambda (\mu m)$')
        plt.ylabel(r'$Insertion loss (dB)$')
        plt.ylim(-60, 0)
        if filename:

            plt.savefig('output/WDMs&loss/WDM_dB_high_' +
                        str(self.l1)+'_low_'+str(self.l2)+'.png')
        else:
            plt.show()
        plt.close(fig)
        return None


def create_file_structure(kk=''):
    """
    Is set to create and destroy the filestructure needed 
    to run the program so that the files are not needed in the repo
    """
    folders_large = ('output_dump',
                     'output_final', 'output'+str(kk))
    folders_large += (folders_large[-1] + '/output',)
    folders_large += (folders_large[-1] + '/data',)
    folders_large += (folders_large[-2] + '/figures',)

    outs = folders_large[-1]
    folders_figures = ('/frequency', '/time', '/wavelength')
    for i in folders_figures:
        folders_figures += (i+'/portA', i+'/portB')
    for i in folders_figures:
        folders_large += (outs + i,)
    folders_large += (outs+'/WDMs',)
    for i in folders_large:
        if not os.path.isdir(i):
            os.system('mkdir ' + i)
    return None


class Splicer(WDM):

    def __init__(self, fopa=False, loss=1):
        self.loss = loss
        self.c1 = 10**(-0.1*self.loss/2.)
        self.c2 = (1 - 10**(-0.1*self.loss))**0.5
        if fopa:
            self.U_calc = self.U_calc_over

    def U_calc_over(self, U_in):
        return U_in

    def U_calc(self, U_in):
        """
        Operates like a beam splitter that 
        reduces the optical power by the loss given (in dB).
        """
        U_out1 = U_in[0] * self.c1 + 1j * U_in[1] * self.c2
        U_out2 = 1j * U_in[0] * self.c2 + U_in[1] * self.c1
        return U_out1, U_out2


class Maintain_noise_floor(object):

    def pass_through(u, noisef):
        U = fftshift(fft(u), axis = -1)
        U  = U +  U - noise_f
        u = ifft(fftshift(U, axis = -1))
        return u 

def norm_const(u, sim_wind):
    t = sim_wind.t
    fv = sim_wind.fv
    U_temp = fftshift(fft(u), axes=-1)
    first_int = simps(np.abs(U_temp)**2, fv)
    second_int = simps(np.abs(u)**2, t)
    return (first_int/second_int)**0.5


class Noise(object):

    def __init__(self, int_fwm, sim_wind):
        self.pquant = np.sum(
            1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))

        self.pquant = (self.pquant/2)**0.5
        self.pquant_f = np.mean(
            np.abs(self.noise_func_freq(int_fwm, sim_wind))**2)
        return None

    def noise_func(self, int_fwm):
        seed = np.random.seed(int(time()*np.random.rand()))
        noise = self.pquant * (np.random.randn(int_fwm.nm, int_fwm.nt)
                               + 1j*np.random.randn(int_fwm.nm, int_fwm.nt))
        return noise

    def noise_func_freq(self, int_fwm, sim_wind):
        self.noise = self.noise_func(int_fwm)
        noise_freq = fftshift(fft(self.noise), axes=-1)
        return noise_freq


@profile
def pulse_propagation(u, U, int_fwm, M1, M2, Q, sim_wind, hf,
                     Dop, dAdzmm, gam_no_aeff):
    """Pulse propagation part of the code. We use the split-step fourier method
       with a modified step using the RK45 algorithm. 
    """
    dztot = 0  # total distance traveled
    Safety = 0.95
    u1 = u[:, :]
    dz = int_fwm.dz * 1
    exitt = False
    while not(exitt):
        # trick to do the first iteration
        delta = 2*int_fwm.maxerr
        while delta > int_fwm.maxerr:
            u1new = ifft(np.exp(Dop*dz/2)*fft(u1))
            #print('dz {}'.format(dz))
            A, delta = RK45CK(dAdzmm, u1new, dz, M1, M2, Q, sim_wind.tsh,
                              sim_wind.dt, hf, sim_wind.w_tiled, gam_no_aeff)
            if (delta > int_fwm.maxerr):
                # calculate the step (shorter) to redo
                dz *= Safety*(int_fwm.maxerr/delta)**0.25
        ###############Successful step###############
        # propagate the remaining half step
        u1 = ifft(np.exp(Dop*dz/2)*fft(A))
        dztot += dz
        # update the propagated distance

        

        if delta == 0:
            dz = Safety*int_fwm.dzstep
        else:
            try:
                dz = np.min(
                    [Safety*dz*(int_fwm.maxerr/delta)**0.2,
                     Safety*int_fwm.dzstep])
            except RuntimeWarning:
                dz = Safety*int_fwm.dzstep
        ###################################################################

        if dztot == (int_fwm.dzstep):
            exitt = True
        elif ((dztot + dz) >= int_fwm.dzstep):
            dz = int_fwm.dzstep - dztot
        ###################################################################
    u = u1
    U = fftshift(fft(u), axes=-1)
    int_fwm.dz = dz*1
    return u, U


def dbm_nm(U, sim_wind, int_fwm):
    """
    Converts The units of freequency to units of dBm/nm
    """
    U_out = U / sim_wind.T**2
    U_out = -1*w2dbm(U_out)
    dlv = [sim_wind.lv[i+1] - sim_wind.lv[i]
           for i in range(len(sim_wind.lv) - 1)]
    dlv = np.asanyarray(dlv)
    for i in range(int_fwm.nm):
        U_out[:, i] /= dlv[i]
    return U_out


def fv_creator(lam_p1, lams, P_s, Deltaf, int_fwm):
    """
    Creates the freequency grid of the simmualtion and returns it.. 
    """
    fp = 1e-3*c / lam_p1
    fs = 1e-3*c / lams
    fv1 = np.linspace(fp - Deltaf, fp, int_fwm.nt//2)
    df = fv1[1] - fv1[0]
    fv2 = [fp+df]
    for i in range(1, int_fwm.nt//2):
        fv2.append(fv2[i - 1]+df)
    fv2 = np.array(fv2)
    fv = np.concatenate((fv1, fv2))
    p_pos = np.where(fv == fp)[0][0]
    where = [p_pos]
    if P_s != 0:

        s_pos = np.argmin(np.abs(fv - fs))

        where.append(s_pos)

    else:
        where.append(0)
    check_ft_grid(fv, df)
    return fv, where


def energy_conservation(entot):
    if not(np.allclose(entot, entot[0])):
        fig = plt.figure()
        plt.plot(entot)
        plt.grid()
        plt.xlabel("nplots(snapshots)", fontsize=18)
        plt.ylabel("Total energy", fontsize=18)
        plt.close()
        sys.exit("energy is not conserved")
    return 0


def check_ft_grid(fv, diff):
    """Grid check for fft optimisation"""
    if np.log2(np.shape(fv)[0]) == int(np.log2(np.shape(fv)[0])):
        nt = np.shape(fv)[0]
    else:
        print("fix the grid for optimization  \
             of the fft's, grid:" + str(np.shape(fv)[0]))
        sys.exit(1)

    lvio = []
    for i in range(len(fv)-1):
        lvio.append(fv[i+1] - fv[i])

    grid_error = np.abs(np.asanyarray(lvio)[:]) - np.abs(diff)
    if not(np.allclose(grid_error, 0, rtol=0, atol=1e-12)):
        print(np.max(grid_error))
        sys.exit("your grid is not uniform")
    return 0


class create_destroy(object):
    """
    creates and destroys temp folder that is used for computation. Both methods needs to be run
    before you initiate a new variable
    """

    def __init__(self, variable, pump_wave=''):
        self.variable = variable
        self.pump_wave = pump_wave
        return None

    def cleanup_folder(self):
        # for i in range(len(self.variable)):
        os.system('mv output'+self.pump_wave + ' output_dump/')
        return None

    def prepare_folder(self):
        for i in range(len(self.variable)):
            os.system('cp -r output'+self.pump_wave +
                      '/output/ output'+self.pump_wave+'/output'+str(i))
        return None


def power_idler(spec, fv, sim_wind, fv_id):
    """
    Set to calculate the power of the idler. The possitions
    at what you call an idler are given in fv_id
    spec: the spectrum in freequency domain
    fv: the freequency vector
    T: time window
    fv_id: tuple of the starting and
    ending index at which the idler is calculated
    """
    E_out = simps((sim_wind.t[1] - sim_wind.t[0])**2 *
                  np.abs(spec[fv_id[0]:fv_id[1], 0])**2, fv[fv_id[0]:fv_id[1]])
    P_bef = E_out/(2*np.max(sim_wind.t))
    return P_bef


class birfeg_variation(object):

    def __init__(self, Da):
        self._P_mat(Da)
        return None

    def _P_mat(self, Da):
        self.P = np.array([[np.cos(Da), 1j * np.sin(Da)],
                           [1j * np.sin(Da), np.cos(Da)]])
        return None

    def bire_pass(self, u, i):
        u1_temp, u2_temp = u[0, :]*1, u[1, :]*1
        try:
            u[0, :] = self.P[0, 0][i] * u1_temp + \
                self.P[0, 1][i] * u2_temp
            u[1, :] = self.P[1, 0][i] * u1_temp + \
                self.P[1, 1][i] * u2_temp
        except IndexError:
            pass
        return u
