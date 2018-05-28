import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c
import h5py
import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
font = {'size': 18}

mpl.rc('font', **font)


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


class Plotter_saver(object):

    def __init__(self, plots, filesaves):
        if plots and filesaves:
            self.exporter = self.plotter_saver_both
        elif plots and not(filesaves):
            self.exporter = self.plotter_only
        elif not(plots) and filesaves:
            self.exporter = self.saver_only
        else:
            sys.exit("You are not exporting anything,\
    				  wasted calculation")
        return None

    def plotter_saver_both(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                           f_p, f_s, which, ro, mode_names, pump_wave='',
                           filename=None, title=None, im=0, plots=True):
        self.plotter(index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, which, ro, mode_names, pump_wave,
                     filename, title, im, plots)
        self.saver(index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s,
                   which, ro, mode_names, pump_wave, filename, title,
                   im, plots)
        return None

    def plotter_only(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, which, ro, mode_names, pump_wave='',
                     filename=None, title=None, im=0, plots=True):
        self.plotter(index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, which, ro, mode_names, pump_wave,
                     filename, title, im, plots)
        return None

    def saver_only(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                   f_p, f_s, which, ro, mode_names, pump_wave='',
                   filename=None, title=None, im=0, plots=True):
        self.saver(index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s,
                   which, ro, mode_names, pump_wave, filename, title,
                   im, plots)
        return None

    def plotter(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                f_p, f_s, which, ro, mode_names, pump_wave='',
                filename=None, title=None, im=0, plots=True):
        """Plots many modes"""

        x, y = 1e-3*c/sim_wind.fv, w2dbm(np.abs(U[:, :])**2)
        xlim, ylim = [900, 1250], [-80, 100]
        xlabel, ylabel = r'$\lambda (nm)$', r'$Spectrum (a.u.)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/wavelength/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, which, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)

        # Frequency
        x, y = sim_wind.fv, w2dbm(np.abs(U[:, :])**2)
        xlim, ylim = [np.min(x), np.max(x)], [-20, 120]
        xlabel, ylabel = r'$f (THz)$', r'$Spectrum (a.u.)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/frequency/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, which, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)

        # Time
        x, y = sim_wind.t, np.abs(u[:, :])**2
        xlim, ylim = [np.min(x), np.max(x)], [np.min(y), np.max(y)]
        xlabel, ylabel = r'$\lambda (nm)$', r'$Spectrum (W)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/time/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, which, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)
        return None

    def saver(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s,
              which, ro, mode_names, pump_wave='', filename=None, title=None,
              im=0, plots=True):
        """Dump to HDF5 for postproc"""

        if filename[:4] != 'port':
            layer = filename[-1]+'/'+filename[:-1]
        else:
            layer = filename
        extra_data = np.array([int_fwm.tot_z, which, int_fwm.nm,P0_p, P0_s, f_p, f_s, ro])

        try:
            save_variables('data_large', str(layer), filepath='output'+pump_wave+'/output'+str(index)+'/data/', U=np.abs(U[:, :]), t=sim_wind.t,
                           fv=sim_wind.fv, lv=sim_wind.lv, extra_data = extra_data)
        except RuntimeError:
            os.system('rm output'+pump_wave+'/output' +
                      str(index)+'/data/data_large.hdf5')
            save_variables('data_large', layer, filepath='output'+pump_wave+'/output'+str(index)+'/data/', U=np.abs(U[:, :]), t=sim_wind.t, 
                           fv=sim_wind.fv, lv=sim_wind.lv, extra_data = extra_data)
            pass
        return None


def plot_multiple_modes(nm, x, y, which, mode_names, ylim, xlim, xlabel, ylabel, title, filesave=None, im=None):
    """
    Dynamically plots what is asked of it for multiple modes given at set point.
    """
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.subplots_adjust(hspace=0.1)
    for i, v in enumerate(range(nm)):
        v = v+1
        ax1 = plt.subplot(nm, 1, v)
        plt.plot(x, y[i, :], '-', label=mode_names[i])
        ax1.legend(loc=2)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        if i != nm - 1:
            ax1.get_xaxis().set_visible(False)
    ax = fig.add_subplot(111, frameon=False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(title)
    plt.grid()
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if type(im) != int:
        newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
        newax.imshow(im)
        newax.axis('off')
    if filesave == None:
        plt.show()
    else:
        plt.savefig(filesave, bbox_inched='tight')
    plt.close(fig)
    return None


def animator_pdf_maker(rounds, pump_index):
    """
    Creates the animation and pdf of the FOPO at different parts of the FOPO 
    using convert from imagemagic. Also removes the pngs so be carefull

    """
    print("making pdf's and animations.")
    space = ('wavelength', 'freequency', 'time')
    for sp in space:
        file_loc = 'output/output'+str(pump_index)+'/figures/'+sp+'/'
        strings_large = ['convert '+file_loc+'00.png ']
        for i in range(4):
            strings_large.append('convert ')
        for ro in range(rounds):
            for i in range(4):
                strings_large[i+1] += file_loc+str(ro)+str(i+1)+'.png '
            for w in range(1, 4):
                if i == 5:
                    break
                strings_large[0] += file_loc+str(ro)+str(w)+'.png '
        for i in range(4):
            os.system(strings_large[i]+file_loc+str(i)+'.pdf')

        file_loca = file_loc+'portA/'
        file_locb = file_loc+'portB/'
        string_porta = 'convert '
        string_portb = 'convert '
        for i in range(rounds):
            string_porta += file_loca + str(i) + '.png '
            string_portb += file_locb + str(i) + '.png '

        string_porta += file_loca+'porta.pdf '
        string_portb += file_locb+'portb.pdf '
        os.system(string_porta)
        os.system(string_portb)

        for i in range(4):
            os.system(
                'convert -delay 30 '+file_loc+str(i)+'.pdf '+file_loc+str(i)+'.mp4')
        os.system('convert -delay 30 ' + file_loca +
                  'porta.pdf ' + file_loca+'porta.mp4 ')
        os.system('convert -delay 30 ' + file_locb +
                  'portb.pdf ' + file_locb+'portb.mp4 ')

        for i in (file_loc, file_loca, file_locb):
            print('rm ' + i + '*.png')
            os.system('rm ' + i + '*.png')
        os.system('sleep 5')
    return None


def read_variables(filename, layer, filepath=''):
    with h5py.File(filepath+str(filename)+'.hdf5', 'r') as f:
        D = {}
        for i in f.get(layer).keys():
            try:
                D[str(i)] = f.get(layer + '/' + str(i)).value
            except AttributeError:
                pass
    return D


def save_variables(filename, layers, filepath='', **variables):

    with h5py.File(filepath + filename + '.hdf5', 'a') as f:
        for i in (variables):
            f.create_dataset(layers+'/'+str(i), data=variables[i])
    return None
