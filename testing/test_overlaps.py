import sys
sys.path.append('src')
import overlaps
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c, pi
import os

def test_loader():
	initial = overlaps.fibre_overlaps_loader()
	
	os.system('rm loading_data/M1_M2_new_2m.hdf5')
	
	overlaps.main()
	ending = overlaps.fibre_overlaps_loader()
	for i, j in zip(initial,ending):
		assert_allclose(i,j)