import numpy as np
from scipy.integrate import dblquad
from scipy.io import loadmat
import h5py
import os


def field0(y, x, w):
    return np.exp(-(x**2+y**2)/w**2)


def field1(y, x, w):
    return (2*2**0.5*x/w)*np.exp(-(x**2+y**2)/w**2)


def calc_overlaps():
    overlaps = np.zeros([2, 8])
    neff1 = 1
    neff2 = 1
    n0 = 1
    r = 62.45  # radius of the fibre in microns
    # for Q0000
    overlaps[:, 0] = (n0/neff1)**2*1/161.  # top/bottom

    # for Q1111
    overlaps[:, -1] = (n0/neff2)**2*1/170.  # top/bottom
    # for Qeven
    # load widths in meters from previous calculations
    widths = np.loadtxt('loading_data/widths.dat')
    # Convert to microns so we do the integrations in microns
    w = widths[0:2]*1e6

    def int1(y, x): return field1(y, x, w[1])**2*field0(y, x, w[0])**2

    def int2(y, x): return field0(y, x, w[0])**2

    def int3(y, x): return field1(y, x, w[1])**2
    top = dblquad(int1, -r, r, lambda x: -r, lambda x: r)[0]
    bottom = dblquad(int2, -r, r, lambda x: -r,
                     lambda x: r)[0]*dblquad(int3, -r, r,
                                             lambda x: -r, lambda x: r)[0]

    overlaps[:, 1:7] = (n0**2/(neff1*neff2))*top/bottom
    overlaps *= 1e12
    overlaps /= 3
    return overlaps



def fibre_overlaps_loader(filepath='loading_data'):
    """
    Loads, or calculates if not there, the M1, M2 and Q matrixes. 
    """
    overlap_file = os.path.join(filepath, 'M1_M2_new_2m.hdf5')
    

    if os.path.isfile(overlap_file):
        keys = ('M1', 'M2', 'Q')
        data = []
        with h5py.File(overlap_file, 'r') as f:
            for i in keys:
                data.append(f.get(str(i)).value)
        data = tuple(data)
    else:
        data = main()
    return data


def save_variables(filename, **variables):

    with h5py.File(filename + '.hdf5', 'a') as f:
        for i in (variables):
            f.create_dataset(str(i), data=variables[i])
    return None


def main():
    Q_matrix = np.real(calc_overlaps())
    mat = loadmat("loading_data/M1_M2_2m.mat", squeeze_me=True)
    M1_load = mat['M1']
    M2_load = mat['M2']

    M2 = np.uint32(M2_load - 1)
    M1 = np.empty([5, 8], dtype=np.uint32)
    M1[0:4, :] = np.int32(np.real(M1_load[0:4, :] - 1))
    M1[4, :] = np.int32(np.real(M1_load[-1, :] - 1))
    D = {'M1': M1, 'M2': M2, 'Q': Q_matrix}
    save_variables('loading_data/M1_M2_new_2m', **D)
    return M1, M2, Q_matrix

if __name__ == '__main__':
    if os.path.isfile('loading_data/M1_M2_new_2m.hdf5'):
        os.system('rm loading_data/M1_M2_new_2m.hdf5')
    main()
