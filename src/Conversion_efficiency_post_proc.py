import numpy as np
import os
import pickle as pl
import tables
import h5py
from scipy.constants import c, pi
import gc
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from data_plotters_animators import read_variables
from functions import *
import warnings 
warnings.filterwarnings('ignore')
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from numpy.fft import fftshift
import scipy
from os import listdir
from joblib import Parallel, delayed
font = {'size'   : 16}
matplotlib.rc('font'
              , **font)


def selmier(l):
    a = 0.6961663*l**2/(l**2 - 0.0684043**2)
    b = 0.4079426*l**2/(l**2 - 0.1162414**2)
    c = 0.8974794*l**2/(l**2 - 9.896161**2)
    return (1 + a + b +c)**0.5





class Conversion_efficiency(object):
    def __init__(self, freq_band_HW, last, safety, possition, filename=None, filepath='',filename2 = 'CE',filepath2 = 'output_final/'):
        self.mode_names = ('LP01', 'LP11a')
        self.n = 1.444
        self.last = last
        self.safety = safety

        P_p1, P_p2, P_s, self.fv, self.lv,self.t, self.where, self.L = self.load_input_param(filepath)
        

        if freq_band_HW is 'df':
            freq_band_HW = 2* (self.fv[1] - self.fv[0])

        self.input_powers = (P_p1, P_p2, P_s)


        self.U_in, self.U_out = self.load_spectrum(possition,filename, filepath)
        #print(self.U_in.shape)
        #print(self.U_out.shape)
        #sys.exit()


        #self.U_in = dbm2w(np.max(w2dbm(self.U_in[0,:])) - w2dbm(self.U_in))
        #self.U_out  = dbm2w(w2dbm(np.max(self.U_out[0,0,:])) - w2dbm(self.U_out))

        if type(last) is float:
            self.last = int(last* self.U_out.shape[1])
        else:
            self.last = last


        self.f_waves  = [self.fv[i] for i in self.where]


        self.nt = np.shape(self.U_in)[-1]
        self.rounds = np.shape(self.U_in)[-2]
        self.possition = possition
        temp = np.argmin(abs(self.fv - 0.5 * (self.fv[0] + self.fv[-1]) - freq_band_HW))
        self.band_idx_seper = [i * temp for i in (-1,1)]


        self.lam_waves = [1e-3*c/i for i in self.f_waves]

        
        self.U_large_norm = np.empty_like(self.U_out)

        n_vec = [selmier(1e-3*i) for i in self.lam_waves]
        self.time_trip = [self.L*n/c for n in n_vec]
        
        
        for i in range(self.U_large_norm.shape[0]):
            self.U_large_norm[i,:,:] =\
                    w2dbm(np.abs(self.U_out[i,:,:])**2)- np.max(w2dbm(np.abs(self.U_in[0,:])**2))



        start_vec = [i - freq_band_HW for i in self.f_waves]
        end_vec =   [i + freq_band_HW for i in self.f_waves]

        
        P_out_vec = np.empty([6,self.U_out.shape[1]])
        count = 0
        for start, end in zip(start_vec[:3], end_vec[:3]):
            start_i = np.argmin(np.abs(self.fv - start))
            end_i = np.argmin(np.abs(self.fv - end))
            for ii,UU in enumerate(self.U_out[0,:,:]):
                self.spec = UU
                P_out_vec[count,ii] = self.calc_P_out(start_i,end_i)
            count +=1
        


        for start, end in zip(start_vec[3:], end_vec[3:]):
            start_i = np.argmin(np.abs(self.fv - start))
            end_i = np.argmin(np.abs(self.fv - end))
            for ii,UU in enumerate(self.U_out[1,:,:]):
                self.spec = UU
                P_out_vec[count,ii] = self.calc_P_out(start_i,end_i)
            count +=1

        start_i = np.argmin(np.abs(self.fv - start_vec[2]))
        end_i = np.argmin(np.abs(self.fv - end_vec[2]))
        
        self.spec = self.U_out[0,0,:]
        self.P_signal_in = self.calc_P_out(start_i,end_i)



        self.P_out_vec = P_out_vec

        
        #for l, la in enumerate(last):
        D_now = {}
        D_now['L'] = self.L
        D_now['P_out'] = np.mean(self.P_out_vec[:,-self.last::], axis = 1)
        

        D_now['CE'] = D_now['P_out']/ self.P_signal_in

        D_now['P_out_std'] = np.std(self.P_out_vec[:,-self.last::], axis = 1)

        D_now['CE_std'] = np.std(self.P_out_vec[:,-self.last::] / self.P_signal_in, axis = 1)

        D_now['rin'] = 10*np.log10(self.time_trip*D_now['P_out_std']**2 / D_now['P_out']**2)
        D_now['input_powers'] = self.input_powers
        D_now['frequencies'] = self.f_waves
        

        for i,j in zip(D_now.keys(), D_now.values()):
            D_now[i] = [j]
        
        if os.path.isfile(filepath2+filename2+'.pickle'):
            with open(filepath2+filename2+'.pickle','rb') as f:
                D = pl.load(f)
            for i,j in zip(D.keys(), D.values()):
                D[i] = j + D_now[i]
        else:
            D = D_now
        with open(filepath2+filename2+'.pickle','wb') as f:
            pl.dump(D,f)
        self.input_data_formating()
        return None
    
    def load_input_param(self, filepath=''):
        filename='input_data'
        
        D = read_variables(filepath+filename, '0/0')
        P_p1 = D['P_p1']
        P_p2 = D['P_p2']
        P_s = D['P_s']
        fv = D['fv']
        lv = D['lv']
        t = D['t']
        where = D['where']
        L = D['L']
        return P_p1, P_p2, P_s, fv, lv, t, where, L

    
    def load_spectrum(self, possition,filename='data_large', filepath=''):


        U_in = read_variables(filepath + filename, 'input')['U']
        U_out = read_variables(filepath + filename, 'results/'+possition)['U']


        U_in, U_out = (np.abs(i)**2 * (self.t[1] - self.t[0])**2 for i in (U_in, U_out))

        return U_in,U_out
    
    def calc_P_out(self,i,j):
       
        P_out = simps(self.spec[i:j],\
                     self.fv[i:j])/(2*np.max(self.t))
        return P_out   
    

    def input_data_formating(self):
        unstr_str = r"$P_1=$ {0:.2f} W, $P_s=$ {1:.2f} mW, "+\
        r"$P_2=$ {2:.2f} W, "+\
        r"$\lambda_1=$ {3:.2f} nm, $\lambda_s=$ {4:.2f} nm, "+\
        r"$\lambda_2=$ {5:.2f} nm,"
        input_data = (self.input_powers[0], self.input_powers[2]*1e3, self.input_powers[1],
                                            self.lam_waves[1], self.lam_waves[2], self.lam_waves[4])

        self.title_string = unstr_str.format(*input_data)
        input_data_str = ('P_1', 'P_s', 'P_2', 'l1', 'ls', 'l2')
        self._data ={i:j for i, j in zip(input_data_str,input_data)}
        return None


    def P_out_round(self,P,CE,filepath,filesave):
        """Plots the output average power with respect to round trip number"""
        x = range(P.shape[-1])
        y = np.asanyarray(P)
        names = ('MI', 'P1', 'S', 'PC', 'P2', 'BS')
       

        for i, name in enumerate(names):
            fig = plt.figure(figsize=(20.0, 10.0))
            plt.subplots_adjust(hspace=0.1)
            
            plt.plot(x, y[i,:], '-')
            plt.title(CE.title_string)
            plt.grid()
            plt.xlabel('Rounds')
            plt.ylabel('Output Power (W)')
            plt.savefig(filepath+'/'+name+'/power_per_round'+filesave+'.png')
            data = (range(len(P)), P)
            
            with open(filepath+'/'+name+'/power_per_round'+filesave+'.pickle','wb') as f:
                pl.dump((data,CE._data),f)
            plt.clf()
            plt.close('all')
        return None


    def final_1D_spec(self,ii,CE,filename,wavelengths = None):
        x,y = self.fv, self.U_large_norm[:,-1,:]
        fig = plt.figure(figsize=(20.0, 10.0))
        for i in range(y.shape[0]):

            plt.plot(x, y[i, :], '-', label=self.mode_names[i])
        plt.legend(loc = 2)


        plt.title(self.title_string)
        plt.grid()
        plt.xlabel(r'$f (THz)$')
        plt.ylabel(r'Spec (dB)')
        #plt.ylim([-200,1])
        #plt.xlim([192, 194.5])
        plt.savefig(filename+'.png', bbox_inches = 'tight')


        data = (x, y)
        #with open(filename+str(ii)+'.pickle','wb') as f:
        #    pl.dump((data,CE._data),f)
        plt.clf()
        plt.close('all')
        return None


def plot_rin(var,var2 = 'rin',filename = 'CE', filepath='output_final/', filesave= None):
    var_val, CE,std = read_CE_table(filename,var,var2 = var2,file_path=filepath)
    std = std[var2].as_matrix()
    if var is 'arb':
        var_val = [i for i in range(len(CE))] 
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.plot(var_val, 10*np.log10(CE),'o-')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(var)
    plt.ylabel('RIN (dBc/hz)')
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    data = (var_val, CE,std)
    with open(str(filesave)+'.pickle','wb') as f:
        pl.dump((fig,data),f)
    plt.clf()
    plt.close('all')

    return None

def read_CE_table(x_key,y_key ,filename, std = False, decibels = False):
    with open(filename+'.pickle','rb') as f:
        D = pl.load(f)
    

    D['P1'], D['P2'], D['Ps'] = [[D['input_powers'][i][j] for i in range(len(D['input_powers']))] for j in range(3)]

    D['f_mi'], D['f_p1'], D['f_s'],\
              D['f_pc'], D['f_p2'],\
                        D['f_bs'] =\
                        [[D['frequencies'][i][j] for i in range(len(D['frequencies']))] for j in range(6)]
    
    D['l_mi'], D['l_p1'], D['l_s'],\
              D['l_pc'], D['l_p2'],\
                        D['l_bs'] =\
                        [[1e-3*c/D['frequencies'][i][j] for i in range(len(D['frequencies']))] for j in range(6)]
    

    D['CE_mi'], D['CE_p1'], D['CE_s'],\
              D['CE_pc'], D['CE_p2'],\
                        D['CE_bs'] =\
                        [[D['CE'][i][j] for i in range(len(D['CE']))] for j in range(6)]
    
    D['CEstd_mi'], D['CEstd_p1'], D['CEstd_s'],\
          D['CEstd_pc'], D['CEstd_p2'],\
                    D['CEstd_bs'] =\
                    [[D['CE_std'][i][j] for i in range(len(D['CE_std']))] for j in range(6)]


    D['P_out_mi'], D['P_out_p1'], D['P_out_s'],\
              D['P_out_pc'], D['P_out_p2'],\
                        D['P_out_bs'] =\
                        [[D['P_out'][i][j] for i in range(len(D['P_out']))] for j in range(6)]
    
    D['P_out_std_mi'], D['P_out_std_p1'], D['P_out_std_s'],\
          D['P_out_std_pc'], D['P_out_std_p2'],\
                    D['P_out_std_bs'] =\
                    [[D['P_out_std'][i][j] for i in range(len(D['P_out_std']))] for j in range(6)]
    
    D['rin_mi'], D['rin_p1'], D['rin_s'],\
      D['rin_pc'], D['rin_p2'],\
                D['rin_bs'] =\
                [[D['rin'][i][j] for i in range(len(D['rin']))] for j in range(6)]             


    x = D[x_key]
    if y_key[:2] == 'CE' and decibels == True:
        y = 10 * np.log10(np.asarray(D[y_key]))
    else:
        y = np.asarray(D[y_key])

    if std:
        try:
            err_bars = D['y_key'+'_std']
        except KeyError:
            sys.exit('There is not error bar for the variable you are asking for.')
    else:
        err_bars = 0
    x,y,err_bars = np.asanyarray(x),np.asanyarray(y), np.asanyarray(y)
    return x,y,err_bars


def plot_CE(x_key,y_key,std = True, decibels = False,filename = 'CE', filepath='output_final/', filesave= None):
    
    x, y, err_bars = read_CE_table(x_key,y_key,filepath+filename,decibels = decibels,std = std )




    fig = plt.figure(figsize=(20.0, 10.0))
    plt.subplots_adjust(hspace=0.1)
    mode_labels = ('LP01x','LP01y')


    
    if std:
        plt.errorbar(x,y, yerr=err_bars, capsize= 10)
    else:
        plt.plot(x - 1549,y)
    
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid()
    plt.xticks(np.arange(0, 2.75, 0.25))


    


    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    data = (x, y,err_bars)
    with open(str(filesave)+'.pickle','wb') as f:
        pl.dump(data,f)
    plt.clf()
    plt.close('all')
    return None



def contor_plot(CE,fmin = None,fmax = None,  rounds = None,folder = None,filename = None):


    if rounds is None:
        rounds = np.shape(CE.U_large_norm)[0]
   
    CE.ro = range(rounds)
    x,y = np.meshgrid(CE.ro[:rounds], CE.fv[:])
    z = CE.U_large_norm[:rounds,:,:]
    
    low_values_indices = z < -60  # Where values are low
    z[low_values_indices] = -60  # All low values set to 0
    for nm in range(z.shape[1]):
        fig = plt.figure(figsize=(20,10))
        plt.contourf(x,y, z[:,nm,:].T, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.jet)
        plt.xlabel(r'$rounds$')
        plt.ylim(fmin,fmax)
        plt.ylabel(r'$f(THz)$')
        plt.colorbar()
        plt.title(CE.title_string)
        data = (CE.ro, CE.fv, z)
        if filename is not None:
            plt.savefig(folder+str(nm)+'_'+filename, bbox_inches = 'tight')
            plt.clf()
            plt.close('all')
        else:
            plt.show()

    if filename is not None:
        with open(str(folder+filename)+'.pickle','wb') as f:
            pl.dump((data,CE._data),f)
    return None


def P_out_round_anim(CE,iii,filesave):
    """Plots the output average power with respect to round trip number"""
    tempy = CE.P_out_vec[:iii]
    
    fig = plt.figure(figsize=(7,1.5))
    plt.plot(range(len(tempy)), tempy)
    plt.xlabel('Oscillations')
    plt.ylabel('Power')
    plt.ylim(0,np.max(CE.P_out_vec)+0.1*np.max(CE.P_out_vec))
    plt.xlim(0,len(CE.P_out_vec))
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    plt.close('all')
    plt.clf()
    return None


def tick_function(X):
    l = 1e-3*c/X
    return ["%.2f" % z for z in l]



def main2(ii):
    ii = str(ii)
    which = which_l+ ii
    
    os.system('mkdir output_final/'+str(ii))
    os.system('mkdir output_final/'+str(ii)+'/pos'+pos+'/ ;'+'mkdir output_final/'+str(ii)+'/pos'+pos+'/many ;'+'mkdir output_final/'+str(ii)+'/pos'+pos+'/spectra;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/BS;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/PC;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/MI;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/P1;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/P2;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers/S;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/casc_powers;'
             +'mkdir output_final/'+str(ii)+'/pos'+pos+'/final_specs;')


    for i in inside_vec[int(ii)]:
        print(ii,i)
        CE = Conversion_efficiency(freq_band_HW = 'df',possition = pos,last = 0.5,\
            safety = 2, filename = 'data_large',\
            filepath = which+'/output'+str(i)+'/data/',filepath2 = 'output_final/'+str(ii)+'/pos'+str(pos)+'/')

        fmin,fmax,rounds  = 310,330,2000#np.min(CE.fv),np.max(CE.fv),None
        fmin,fmax,rounds = None,None, None

        #if CE.U_large_norm.shape[0]>1:
        #    contor_plot(CE,fmin,fmax,rounds,folder = 'output_final/'+str(ii)+'/pos'+pos+'/spectra/',filename= str(ii)+'_'+str(i))
        #contor_plot_time(CE, rounds = None,filename = 'output_final/'+str(ii)+'/pos'+pos+'/'+'time_'+str(ii)+'_'+str(i))
        CE.P_out_round(CE.P_out_vec,CE,filepath =  'output_final/'+str(ii)+'/pos'+pos+'/powers/', filesave =str(ii)+'_'+str(i))
        CE.final_1D_spec(ii,CE,filename = 'output_final/'+str(ii)+'/pos'+pos+'/final_specs/'+'spectrum_fopo_final'+str(i),wavelengths = wavelengths)
        del CE
        gc.collect()
    for x_key,y_key,std, decibel in (('l_s', 'P_out_bs',False, False), ('l_s', 'CE_bs',False, True), ('l_s', 'rin_bs',False, False),\
                            ('l_s', 'P_out_pc',False, False), ('l_s', 'CE_pc',False, True), ('l_s', 'rin_pc',False, False)):
        plot_CE(x_key,y_key,std = std,decibels = decibel, filename = 'CE',\
            filepath='output_final/'+str(ii)+'/pos'+pos+'/', filesave = 'output_final/'+str(ii)+'/pos'+pos+'/many/'+y_key+str(ii))
    return None



#from os.path import , join
data_dump =  'output_dump'
outside_dirs = [f for f in listdir(data_dump)]
inside_dirs = [[f for f in listdir(data_dump+ '/'+out_dir)] for out_dir in outside_dirs ]


which = 'output_dump_pump_wavelengths/7w'
which = 'output_dump_pump_wavelengths/wrong'
which = 'output_dump_pump_wavelengths'
#which = 'output_dump_pump_wavelengths/2_rounds'
#which ='output_dump_pump_powers/ram0ss0'
#which = 'output_dump/'#_pump_powers'
which_l = 'output_dump/output'



outside_vec = range(len(outside_dirs))
#outside_vec = range(2,3)
inside_vec = [range(len(inside) - 1) for inside in inside_dirs]
#inside_vec = [13]
animators = False
spots = range(0,8100,100)
wavelengths = [1200,1400,1050,930,800]
#wavelengths = None


os.system('rm -r output_final ; mkdir output_final')
for pos in ('2', '4'):

    #for ii in outside_vec:
    A = Parallel(n_jobs=6)(delayed(main2)(ii) for ii in outside_vec)

        
    #os.system('rm -r prev_anim/*; mv animators* prev_anim')

