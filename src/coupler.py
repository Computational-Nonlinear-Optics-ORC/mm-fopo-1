import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
from scipy.special import jv, kv
import time
from scipy.optimize import brenth
from scipy.integrate import simps
import sys


def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))


def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))


class Eigensolver(object):
    def __init__(self, lmin, lmax, N):
        self.l_vec = np.linspace(lmin, lmax, N)

        self._A_ = {'sio2': [0.6965325, 0.0660932**2],
                    'ge': [0.7083925, 0.0853842**2]}
        self._B_ = {'sio2': [0.4083099, 0.1181101**2],
                    'ge': [0.4203993, 0.1024839**2]}
        self._C_ = {'sio2': [0.8968766, 9.896160**2],
                    'ge': [0.8663412, 9.896175**2]}

    def V_func(self, r):
        self.V_vec = r * (2 * pi / self.l_vec) * \
            (self.ncore**2 - self.nclad**2)**0.5

    def indexes(self, l, r):
        per_core, per_clad = 'ge', 'sio2'

        self.A = [self._A_[str(per_core)], self._A_[str(per_clad)]]
        self.B = [self._B_[str(per_core)], self._B_[str(per_clad)]]
        self.C = [self._C_[str(per_core)], self._C_[str(per_clad)]]

        return self.sellmeier(l)

    def initialise_fibre(self, r):
        self.ncore, self.nclad = self.indexes(self.l_vec, r)
        self.r = r
        self.V_func(r)
        print('Maximum V number in space is {}'.format(np.max(self.V_vec)))

    def sellmeier(self, l):
        l = (l*1e6)**2
        n = []
        for a, b, c in zip(self.A, self.B, self.C):
            n.append(
                (1 + l*(a[0]/(l - a[1]) + b[0] /
                        (l - b[1]) + c[0]/(l - c[1])))**0.5)
        return n

    def equation_01(self, u, i):
        w = self.w_equation(u, i)
        return jv(0, u) / (u * jv(1, u)) - kv(0, w) / (w * kv(1, w))

    def equation_11(self, u, i):
        w = self.w_equation(u, i)
        return jv(1, u) / (u * jv(0, u)) + kv(1, w) / (w * kv(0, w))

    def w_equation(self, u, i):
        return (self.V_vec[i]**2 - u**2)**0.5

    def solve_01(self, i):
        return self.solve_equation(self.equation_01, i)

    def solve_11(self, i):
        return self.solve_equation(self.equation_11, i)

    def solve_equation(self, equation, i):
        margin = 1e-15
        m = margin
        s = []
        count = 8
        N_points = 2**count
        found_all = 2

        while found_all < 5:
            nm = len(s)
            u_vec = np.linspace(margin, self.V_vec[i] - margin, N_points)
            eq = equation(u_vec, i)
            s = np.where(np.sign(eq[:-1]) != np.sign(eq[1:]))[0] + 1
            count += 1
            N_points = 2**count
            if nm == len(s):
                found_all += 1
        u_sol, w_sol = np.zeros(len(s)), np.zeros(len(s))

        for iss, ss in enumerate(s):

            Rr = brenth(equation, u_vec[ss-1], u_vec[ss],
                        args=(i), full_output=True)
            # print(Rr)
            u_sol[iss] = Rr[0]
            w_sol[iss] = self.w_equation(Rr[0], i)

        if len(s) != 0:
            return u_sol, w_sol, self.neff(u_sol, w_sol, i)
        else:
            print('-No solutions found for some inputs-')
            print(' V = ', self.V_vec[i])
            print(' R = ', self.r)
            print(' l = ', self.l_vec[i])
            print('---------------------------')
            u = np.linspace(1e-6, self.V_vec[i] - 1e-6, 2048)

            e = equation(u, i)
            plt.plot(np.abs(u), e)

            plt.xlim(u.min(), u.max())
            plt.ylim(-10, 10)
            plt.show()
            sys.exit(1)

    def neff(self, u, w, i):
        return (((self.ncore[i] / u)**2 +
                 (self.nclad[i]/w)**2) / (1/u**2 + 1/w**2))**0.5


def integrate(x, y, E):
    """
    Integrates twice using Simpsons rule from scipy
    to allow 2D  integration.
    """
    return simps(simps(E, y), x)


class Modes(object):
    def __init__(self, a, u_vec, w_vec, neff_vec, lam_vec, grid_points, n):
        self.N_points = grid_points
        self.a = a
        self.set_coordinates(a)
        self.u_vec = u_vec
        self.w_vec = w_vec
        self.k_vec = 2 * pi / lam_vec
        self.beta_vec = neff_vec * self.k_vec
        self.n = n

    def wave_indexing(self, i):
        self.u = self.u_vec[i]
        self.w = self.w_vec[i]
        self.k = self.k_vec[i]
        self.beta = self.beta_vec[i]
        self.s = self.n * (1/self.u**2 + 1/self.w**2) /\
            (jv_(self.n, self.u)/(self.u*jv(self.n, self.u))
             + kv_(self.n, self.w)/(self.w*kv(self.n, self.w)))

    def set_coordinates(self, a):
        self.x, self.y = [np.linspace(-3*a, 3*a, self.N_points)
                          for i in range(2)]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = ((self.X)**2 + (self.Y)**2)**0.5
        self.T = np.arctan(self.Y/self.X)
        return None

    def E_r(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]

        print('lenghts')
        print(self.u)
        print(self.beta)
        print(self.n)
        print(r.shape)
        print(self.a)
        print(self.s)
        temp[r0_ind] = -1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    - 0.5*(1 + self.s)*jv(self.n + 1, self.u * r0 / self.a))

        temp[r1_ind] = -1j * self.beta*self.a*jv(self.n, self.u)\
            / (self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               + 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))

        return temp*np.cos(self.n*theta)  # , temp*np.cos(self.n*theta+pi/2)

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
        return temp*np.sin(self.n*theta)

    def E_zeta(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = jv(self.n, self.u*r0/self.a)
        temp[r1_ind] = jv(self.n, self.u) * \
            kv(self.n, self.w*r1/self.a)/kv(self.n, self.w)
        return temp*np.cos(self.n*theta)

    def E_carte(self):
        Er = self.E_r(self.R, self.T)
        Et = self.E_theta(self.R, self.T)
        Ex, Ey, Ez = [], [], []
        Ex = Er * np.cos(self.T) - Et * np.sin(self.T)
        Ey = Er * np.sin(self.T) + Et * np.cos(self.T)
        Ez = self.E_zeta(self.R, self.T)

        return Ex, Ey, Ez

    def E_abs2(self):
        Ex, Ey, Ez = self.E_carte()
        return np.abs(Ex)**2 + np.abs(Ey)**2


class Coupling_coeff(Modes):

    def __init__(self, a, N_points, ncore, nclad, l_vec, neff):
        self.a = a
        self.N_points = N_points
        self.ncore = ncore
        self.nclad = nclad
        self.Deltan = ncore - nclad
        self.l_vec = l_vec
        self.f_vec = c / l_vec
        self.set_coordinates(self.a)
        self.neff = neff

    def fibre_ref(self, d=-1):
        """
        Creates the refractive index of the fibre if d = -1 or
        creates the couplers seperated at a distance of d. 
        """
        if d != -1:
            R1 = ((self.X - (self.a + d/2))**2 + (self.Y)**2)**0.5
            R2 = ((self.X + (self.a + d/2))**2 + (self.Y)**2)**0.5
            rin = np.where(np.logical_or(R1 <= self.a, R2 <= self.a))
        else:
            R = self.R
            rin = np.where(R <= self.a)
        n0 = np.ones([len(self.l_vec), self.N_points, self.N_points])
        n0 = (self.nclad[:, np.newaxis].T*n0.T).T
        temp = np.zeros([self.N_points, self.N_points])
        for i in range(n0.shape[0]):
            temp[rin] = self.Deltan[i]
            n0[i, :, :] += temp[:, :]
        return n0

    def coupled_index(self, d):
        return self.fibre_ref(d)

    def initialise_integrand_vectors(self, d, Eabs2):
        self.int1_vec = Eabs2
        self.int2_vec = (self.coupled_index(d)**2 -
                         self.fibre_ref()**2) * Eabs2
        return None

    def integrals(self):
        return integrate(self.x, self.y, self.int2_vec) / \
            integrate(self.x, self.y, self.int1_vec)


class Coupling_coefficients(object):
    def __init__(self, lmin, lmax, N, a, N_points, d_vec):
        self.lmin = lmin
        self.lmax = lmax
        self.N = N  # number of frequency grid
        self.a = a  # radius of the two fibres considered
        self.N_points = N_points  # grid of the mode functions [X, Y]
        # vector of the distance between the two fibres (to form coupler)
        self.d_vec = d_vec

    def fibre_calculate(self):
        u, w = self.get_eigenvalues()
        print(len(u), len(w))
        self.get_mode_functions(u, w)
        return None

    def get_eigenvalues(self):
        """
        Calculates the eigenvalues of the fibres considered.
        """
        e = Eigensolver(self.lmin, self.lmax, self.N)
        e.initialise_fibre(self.a)
        u_01, w_01, neff01 = [np.zeros(self.N) for i in range(3)]
        u_11, w_11, neff11 = [np.zeros([2, self.N]) for i in range(3)]
        for i in range(self.N):
            u_01[i], w_01[i], neff01[i] = e.solve_01(i)
            u_11[:, i], w_11[:, i], neff11[:, i] = e.solve_11(i)
        self.e = e
        self.neff = neff01, neff11
        return (u_01, u_11), (w_01, w_11)

    def get_mode_functions(self, u, w):
        u_01, u_11 = u
        w_01, w_11 = w
        neff01, neff11 = self.neff
        Eabs01, Eabs11 = [
            np.zeros([self.N, self.N_points, self.N_points]) for i in range(2)]
        print(len(u_01), len(u_11))
        m01 = Modes(self.a, u_01, w_01, neff01, self.e.l_vec, self.N_points, 1)
        m11 = Modes(self.a, u_11, w_11, neff11, self.e.l_vec, self.N_points, 0)

        for i in range(self.N):
            m01.wave_indexing(i)
            m11.wave_indexing(i)
            Eabs01[i, :, :] = m01.E_abs2()
            Eabs11[i, :, :] = m11.E_abs2()

        self.Eabs = Eabs01, Eabs11
        return None

    def create_coupling_coeff(self, d):
        neff01, neff11 = self.neff
        Eabs01, Eabs11 = self.Eabs

        k01, k11 = [np.zeros(self.N) for i in range(2)]

        couple01 = Coupling_coeff(
            self.a, self.N_points, e.ncore, e.nclad, e.l_vec, neff01)
        couple11 = Coupling_coeff(self.a, self.N_points, e.ncore,
                                  e.nclad, e.l_vec, neff11[0, :])
        couple01.initialise_integrand_vectors(d, Eabs01)
        couple11.initialise_integrand_vectors(d, Eabs11)

        k01 = couple01.f_vec * (pi / (neff01 * c)) * couple01.integrals()
        k11 = couple11.f_vec * (pi / (neff11[0, :] * c)) * couple11.integrals()
        return k01, k11, couple01, couple11


def main():
    lmin = 1546e-9
    lmax = 1555e-9
    N_l = 10
    a = 5e-6
    N_points = 1024
    d_vec = [0.05e-6]

    couple_obj = Coupling_coefficients(lmin, lmax, N_l, a, N_points, d_vec)
    couple_obj.fibre_calculate()
    sys.exit()

    k01, k11, couple01 = create_coupling_coeff(lmin, lmax, N_l, a, N_points, d)
    fig = plt.figure()
    plt.plot(couple01.l_vec*1e9, k01, 'o-')
    plt.plot(couple01.l_vec*1e9, k11, 'o-')
    N_l = 2
    k01, k11, couple01 = create_coupling_coeff(lmin, lmax, N_l, a, N_points, d)
    fig = plt.figure()
    plt.plot(couple01.l_vec*1e9, k01, 'o-')
    plt.plot(couple01.l_vec*1e9, k11, 'o-')
    plt.show()

    plt.plot(couple01.l_vec*1e9, k01/k11)
    plt.show()
    n1 = couple01.coupled_index(d)
    plt.contourf(1e6*couple01.X, 1e6*couple01.Y, n1[0, :, :])
    plt.colorbar()
    plt.show()

    #n1 = couple.coupled_index(1e-6)
    #plt.contourf(1e6*couple.X, 1e6*couple.Y, n1[0, :, :])
    # plt.colorbar()
    # plt.show()

    #xc = np.linspace(-a, a, 1024)
    #yc = np.sqrt(-xc**2+a**2)
    # fig=plt.figure(1)
    #plt.contourf(m01.X, m01.Y,np.abs(mm[0])**2)
    #plt.plot(xc, yc, 'black', linewidth=2.0)
    #plt.plot(xc, -yc, 'black', linewidth=2.0)
    # plt.show()

    # fig=plt.figure(1)
    #plt.contourf(m11.X, m11.Y,np.abs(mm[0])**2)
    #plt.plot(xc, yc, 'black', linewidth=2.0)
    #plt.plot(xc, -yc, 'black', linewidth=2.0)
    # plt.show()
    return None


if __name__ == '__main__':
    main()
