import pstats, cProfile
import cython_integrand
import pickle

with open('../../loading_data/profile.pickl', 'rb') as f:
	D = pickle.load(f)

names = ('u0', 'u0_conj', 'M1', 'M2', 'Q', 'tsh', 'dt','hf', 'w_tiled', 'gam_no_aeff') 
u1,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff = D

cProfile.runctx("cython_integrand.dAdzmm_ron_s1_cython(u1,u0_conj, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()