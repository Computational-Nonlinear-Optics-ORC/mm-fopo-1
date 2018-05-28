
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth, root, brentq, bisect
from scipy.constants import c, pi
from scipy.special import jv, kv,jvp,kvp
from scipy.interpolate import interp1d
from scipy.integrate import simps
import sys
import warnings
from pprint import pprint
from math import factorial
from itertools import combinations_with_replacement, permutations
import h5py
import os
from scipy.interpolate import UnivariateSpline
from numba import jit, vectorize
import time
from six.moves import builtins
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func

#@jit
def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))

#@jit
def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))

def save_variables_step(filename,filepath='', **variables):

    file = os.path.join(filepath, filename+'.hdf5')
    if os.path.isfile(file):
        os.system('rm '+file)
    with h5py.File(file, 'a') as f:
        for i in (variables):
            f.create_dataset(str(i), data=variables[i])
    return None



class Fibre(object):
    """
    Fibre class. Set to initiate the functions needed
    for a fibre ( Seilmier equations etc).
    """

    def __init__(self):
        self._A_ = {'sio2': [0.6965325, 0.0660932**2],
                    'ge': [0.7083925, 0.0853842**2]}
        self._B_ = {'sio2': [0.4083099, 0.1181101**2],
                    'ge': [0.4203993, 0.1024839**2]}
        self._C_ = {'sio2': [0.8968766, 9.896160**2],
                    'ge': [0.8663412, 9.896175**2]}
        return None

    def indexes(self, l, r, per, err, plot = False):
        per_core, per_clad = per

        self.A = [self._A_[str(per_core)], self._A_[str(per_clad)]]
        self.B = [self._B_[str(per_core)], self._B_[str(per_clad)]]
        self.C = [self._C_[str(per_core)], self._C_[str(per_clad)]]

        core, clad = self.sellmeier(l)
        N_cores = len(r)
        try:
        
            self.clad = np.repeat(clad[np.newaxis, :], N_cores, axis=0)
            np.random.seed(int(time.time()+10))

            self.core = np.zeros([N_cores, len(l)])
            for i in range(N_cores):
                self.core[i, :] = \
                        err[i]*(core - clad) + core
        except IndexError:
            #print('index')
            self.clad = clad
            self.core = np.random.rand(N_cores)*err*(core - clad) + core
            pass
        if not(plot):
            #print(self.core)
            #print(self.clad)
            try:
                assert((self.core > self.clad).all())
            except AssertionError:
                print(self.core.shape)
                plt.plot(self.core, label = 'core')
                plt.plot(self.clad, label = 'clad')
                plt.show()
            assert((self.core > self.clad).all())
        return self.core, self.clad

    def sellmeier(self, l):
        l = (l*1e6)**2
        n = []
        for a, b, c in zip(self.A, self.B, self.C):
            n.append(
                (1 + l*(a[0]/(l - a[1]) + b[0]/(l - b[1]) + c[0]/(l - c[1])))**0.5)
        return n

    def V_func(self, l_vec, a_vec):
        V_vec = np.empty([len(a_vec), len(l_vec)])
        temp = (2 * pi / l_vec)
        for i, a in enumerate(a_vec):
            V_vec[i, :] = temp * a * \
                (self.core[i, :]**2 - self.clad[i, :]**2)**0.5
        self.V = V_vec
        return None

    def plot_fibre_n(self, l, r, per, err):
        n = {}
        for per in (('ge', 'ge'),('sio2','sio2')):
            nn = self.indexes(l, r, per, err,plot = True)
            n[str(per[0])] = nn[0]
        perc = (20, 30, 40, 50, 60)
        fig = plt.figure()
        for p,nn in n.items():
            #print(p,nn)
            plt.plot(l*1e9, nn[-1,:], label=p)
        plt.xlabel(r'$\lambda (nm)$')
        plt.ylabel(r'$n$')
        plt.legend()
        #plt.show()
        return None

    def beta_dispersions(self,o_vec, i):
        coefs = self.beta_extrapo(o_vec,i)
        betas = np.empty_like(coefs)
        for i, c in enumerate(coefs[::-1]):
            betas[i] = c * factorial(i)
        return betas


@vectorize('float64(float64,float64,float64,float64,float64,float64)')
def res_faster(a,b,r,u,w,n):
    return (a + b) * (a + b*r**2) \
           - n**2 * (1/u**2 + 1/w**2) * (1/u**2 + r**2 / w**2)



class Eigenvalues(Fibre):
    """
    Sets up to solve and solves the eigenvalue equation
    to find the eigenvalues of HE11. Only works on a single mode 
    fibre. Inherits V number function from Fibre class.
    """

    def __init__(self, l, r, ncore, nclad):
        self.core, self.clad = ncore, nclad
        self.V_func(l, r)
        self.a_vec = r 
        self.l_vec  = l
        self.k = 2 * pi * c / l
        self.ratio = self.clad/self.core
        return None
    #@profile
    def w_f(self, u, i, j):
        """
        Equation to get w eigenvalue with respect to guess u, and V. 
        """
        return (self.V[i, j]**2 - u**2)**0.5
    #@profile
    def eq(self, u_vec, i, j, n=1):
        """
        The eigenvalue equation of a single mdoe fibre,
        set by default to find the HE11 mode. 
        """
        u = u_vec
        #@vectorize('float64(float64,float64,float64)')
        w = self.w_f(u, i, j)

        a = jv_(n, u)/(u*jv(n, u))   
        b = kv_(n, w)/(w*kv(n, w))

        res =  res_faster(a,b,self.ratio[i, j],u,w,n)

        return res



    def neff(self, i, j, u, w):
        """
        Calculates the neff of each mode for sorting so the fundemental mode can be found. 
        """
        return (((self.core[i, j]/u)**2 + (self.clad[i, j]/w)**2)
                / (1/u**2 + 1/w**2))**0.5

    def eigen_solver(self, margin, i, j):
        """
        Finds the eigenvalues of the fibre using breth. 
        Inputs:
            margin: A safety margin to save from the Zero division errors 
                    that arrise from the Eigenvalue equation if u,w = 0
        Returns:
        u, w: eigenvalues of the equation, system exists if non are 
              found and the equation is plotted. 
        """
        converged = False
        m = margin
        V_d = []

        nm = -1
        s = []
        count = 8
        N_points = 2**count
        found_all = 0

        while found_all < 2:
            nm = len(s)
            u_vec = np.linspace(margin, self.V[i, j] - margin, N_points)
            eq = self.eq(u_vec, i, j,n = 1)
            s = np.where(np.sign(eq[:-1]) != np.sign(eq[1:]))[0] + 1
            count += 1
            N_points = 2**count
            if nm == len(s):
                found_all +=1
        u_sol, w_sol = np.zeros(len(s)), np.zeros(len(s))

        for iss, ss in enumerate(s):
            
            Rr = brenth(self.eq, u_vec[ss-1], u_vec[ss],
                        args=(i, j), full_output=True)
            u_sol[iss] = Rr[0]
            w_sol[iss] = self.w_f(Rr[0], i, j)

        if len(s) != 0:
            neffs = self.neff(i, j, u_sol, w_sol)
            neffs = np.nan_to_num(neffs)
            indx_fun = np.argmax(neffs)
            #print(len(s), self.V[i,j], neffs[indx_fun])
            #print(u_sol[indx_fun],w_sol[indx_fun], self.V[i,j])
            #if len(s) >1:
            #    #print(s)
            #    plt.plot(u_vec, eq)

            #    plt.axhline(0, color='black')
            #    plt.title(str(1e9*self.l_vec[j]).format('%3') + ', ' +str(self.a_vec[i]).format('%3'))
            #    plt.ylim(-1,1)
            #    plt.show()
            return u_sol[indx_fun], w_sol[indx_fun]
        else:
            print(
                '----------------------------No solutions found for some inputs--------------------')
            print(' V = ', self.V[i,j])
            print(' R = ', self.a_vec[i])
            print(' l = ', self.l_vec[j])
            print(
                '----------------------------------------------------------------------------------')
            u = np.linspace(1e-6, self.V[i, j] - 1e-6, 2048)
            print(self.V)
            e = self.eq(u, i, j)
            plt.plot(np.abs(u), e)

            plt.xlim(u.min(), u.max())
            plt.ylim(-10, 10)
            plt.show()
            sys.exit(1)


class Betas(Fibre):
    """
    Calculates the betas of the fibre mode.
    """

    def __init__(self, u_vec, w_vec, l_vec, o_vec, o, ncore, nclad):
        #self.k = 2*pi/(l_vec)
        self.u = u_vec
        self.w = w_vec
        self.core, self.clad = ncore, nclad
        self.o_vec = o_vec
        self.o = o
        self.o_norm = self.o_vec - self.o
        return None

    def beta_func(self, o_vec,i):
        """
        Calculates and returns the betas of the fibre 
        """

        return ((o_vec*1e12/c)**2*((self.core[i, :]/self.u[i, :])**2 +
                           (self.clad[i, :]/self.w[i, :])**2)/(1/self.u[i, :]**2
                                                               + 1/self.w[i, :]**2))**0.5
        #return (((self.core[i, j]/u)**2 + (self.clad[i, j]/w)**2)
        #        / (1/u**2 + 1/w**2))**0.5
    def beta_extrapo(self,o_vec, i):
        """
        Gets the polyonomial coefficiencts of beta(omega) with the 
        highest order possible. 
        """
        betas = self.beta_func(o_vec,i)
        deg = 30
        fitted = False
        # warnings.warn(Warning())
        with warnings.catch_warnings():
            while not(fitted):
                warnings.filterwarnings('error')
                try:
                    coef = np.polyfit(self.o_norm, betas, deg=deg)
                    fitted = True
                except Warning:
                    deg -= 1
        return coef


class Modes(Fibre):
    """docstring for Modes"""

    def __init__(self, o_vec, o_c, beta_c, u_vec, w_vec, a_vec, N_points, per, err):
        super().__init__()
        self.n = 1
        self.N_points = N_points
        o_vec *= 1e12
        o_c *= 1e12
        o_norm = o_vec - o_c
        #self.coordinates(x, y)
        self.beta_c = beta_c
        # indexes(self, l, r, per, err)
        self.core = self.indexes(2*pi*c/o_c, a_vec, per, err)[0]
        self.neff = self.beta_c / (o_c / c)

        self.u_vec, self.w_vec = np.zeros(u_vec.shape[0]),\
            np.zeros(u_vec.shape[0])
        for i in range(u_vec.shape[0]):
            self.u_vec[i] = interp1d(o_norm, u_vec[i, :], kind='cubic')(0)
            self.w_vec[i] = interp1d(o_norm, w_vec[i, :], kind='cubic')(0)
        self.a_vec = a_vec
        return None

    def set_coordinates(self, a):
        self.x, self.y = np.linspace(-2*a, 2*a, self.N_points),\
                         np.linspace(-2*a, 2*a, self.N_points)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.R = ((self.X)**2 + (self.Y)**2)**0.5
        self.T = np.arctan(self.Y/self.X)
        return None

    def pick_eigens(self, i):
        self.u = self.u_vec[i]
        self.w = self.w_vec[i]
        self.beta = self.beta_c[i]
        self.a = self.a_vec[i]
        self.s = self.n * (1/self.u**2 + 1/self.w**2) /\
            (jv_(self.n, self.u)/(self.u*jv(self.n, self.u))
             + kv_(self.n, self.w)/(self.w*kv(self.n, self.w)))
        return None


    def E_r(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = -1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    - 0.5*(1 + self.s)*jv(self.n + 1, self.u * r0 / self.a))

        temp[r1_ind] = -1j * self.beta*self.a*jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               + 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))

        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)


    def E_theta(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = 1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    + 0.5*(1 + self.s)*jv(self.n+1, self.u * r0 / self.a))

        temp[r1_ind] = 1j * self.beta*self.a * \
            jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               - 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))
        return temp*np.sin(self.n*theta), temp*np.sin(self.n*theta+pi/2)


    def E_zeta(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = jv(self.n, self.u*r0/self.a)
        temp[r1_ind] = jv(self.n, self.u) * \
            kv(self.n, self.w*r1/self.a)/kv(self.n, self.w)
        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    def E_carte(self):
        Er = self.E_r(self.R, self.T)
        Et = self.E_theta(self.R, self.T)
        Ex, Ey, Ez = [], [], []
        for i in range(len(Er)):
            Ex.append(Er[i] * np.cos(self.T) - Et[i] * np.sin(self.T))
            Ey.append(Er[i] * np.sin(self.T) + Et[i] * np.cos(self.T))
        Ez = self.E_zeta(self.R, self.T)

        return (Ex[0], Ey[0], Ez[0]), (Ex[1], Ey[1], Ez[1])
    @profile
    def Q_matrixes(self):
        self.combination_matrix_assembler()
        Q_large = []
        for i,aa in enumerate(self.a_vec):
            self.set_coordinates(aa)
            self.pick_eigens(i)
            E = np.asanyarray(self.E_carte())
            N = self.Normalised_N(E, i)
            Q = np.zeros([2, len(self.M1_top)], dtype=np.complex)
            for j, com in enumerate(self.M1_top):
                d = self.product_4(com, E)

                Q[0, j] = (1/3) * self.core[i]**2 * self.integrate(d[0]) / (N[com[0]]
                                             * N[com[1]]*N[com[2]] * N[com[3]])
                Q[1, j] = (1/3) * self.core[i]**2 * self.integrate(d[1]) / (N[com[0]]
                                             * N[com[1]]*N[com[2]] * N[com[3]])
                #print(Q)
            if i is 0:
                list_to_keep = []
                for i in range(len(self.M1_top)):
                    if Q[0, i] > 1 or Q[1, i] > 1:
                        list_to_keep.append(i)
            Q = Q[:, list_to_keep]
            Q_large.append(Q)
        M1 = np.asanyarray(self.M1_top)
        M1 = M1[list_to_keep].T
        M2 = np.unique(M1[-2:, :].T, axis=0).T
        M1_end = []
        for i, m1 in enumerate(M1[-2:, :].T):
            for j, m2 in enumerate(M2.T):
                if (m1 == m2).all():
                    M1_end.append(j)
                    break

        M1 = np.vstack((M1, np.asanyarray(M1_end)))

        for qq in Q_large:
            assert len(qq[0, :]) == M1.shape[1]
            assert len(qq[1, :]) == M1.shape[1]
        return M1, M2, np.asanyarray(Q_large)

    def product_2(self, E):
        d1 = np.einsum('ijk,ijk->jk', np.conj(E[0, :, :, :]), E[0, :, :, :])
        d2 = np.einsum('ijk,ijk->jk', np.conj(E[1, :, :, :]), E[1, :, :, :])
        return np.abs(d1), np.abs(d2)

    def product_4(self, com, E):
        d1 = np.einsum('ijk,ijk->jk', np.conj(E[com[0], :, :, :]), E[com[1], :, :, :]) *\
            np.einsum('ijk,ijk->jk', E[com[2], :, :,
                                       :], np.conj(E[com[3], :, :, :]))
        d2 = np.einsum('ijk,ijk->jk', np.conj(E[com[0], :, :, :]), np.conj(E[com[3], :, :, :])) *\
            np.einsum('ijk,ijk->jk', E[com[2], :, :, :], E[com[1], :, :, :])
        return d1, d2

    def denom_integral(self):

        return None

    def Normalised_N(self, E, i):
        """
        Calculated the normalised N integrals for the Q's for two modes.
        """
        d = self.product_2(E)
        return (self.neff[i] * self.integrate(d[0]))**0.5,\
            (self.neff[i] * self.integrate(d[1]))**0.5

    def combination_matrix_assembler(self):
        tot = []
        for i in combinations_with_replacement(range(2), 4):
            l = permutations(i)
            for j in l:
                tot.append(j)
        a = list(set(tot))
        a.sort()
        self.M1_top = a

        return None

    def integrate(self, z):
        """
        Integrates twice using Simpsons rule from scipy
        to allow 2D  integration.
        """
        return simps(simps(z, self.y), self.x)
