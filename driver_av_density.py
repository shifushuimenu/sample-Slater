#!/usr/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt 
import time 
import resource

from sample_Slater import sample_FF_GreensFunction
from test_suite import prepare_test_system_finiteT, square_region

(Nsites, beta, mu, MDF, OBDM) = prepare_test_system_finiteT(Nsites=11, beta=5, mu=0.0, potential='random-binary')
#OBDM_tot = np.loadtxt("../test_data/OBDM_L16x16_U7.2mu3.6_attractiveHubbard.dat")
# OBDM = OBDM_tot[np.ix_([102,103],[102,103])]
#OBDM = square_region(OBDM_tot, L_A=12, x0=1, y0=1)
#Nsites = OBDM.shape[0]

G = np.eye(Nsites) - OBDM

generate_Fock_states = sample_FF_GreensFunction(G=G, Nsamples=10000, update_type='low-rank')

av_density = np.zeros(Nsites)
av2_density = np.zeros(Nsites)

ss = 0
start = time.time()
for occ_vector, sign, weight in generate_Fock_states:
    ss += 1
    print(ss, occ_vector, sign, weight)
    print('RAM used=', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    av_density += occ_vector*sign*weight
    av2_density += (occ_vector*weight)**2    
end = time.time()
elapsed_WCT = end-start
print("elapsed WCT:", elapsed_WCT)    

Nsamples = ss
av_density /= float(Nsamples)
av2_density /= float(Nsamples)
sigma_density = np.sqrt(av2_density - av_density**2) / np.sqrt(Nsamples)

# Plot average density
fig, ax1 = plt.subplots(1,1)
ax1.set_title('Average density from snapshots. \n Nsamples = %d, WCT = %10.4f [sec]' % (Nsamples, elapsed_WCT))
ax1.errorbar(np.arange(Nsites), av_density, yerr=sigma_density, fmt='-+', \
        label='average density')       
ax1.plot(np.arange(Nsites), np.diag(OBDM), label='from Green\'s function')
ax1.legend(loc='lower center')
plt.xlabel(r'site $i$')
plt.ylabel(r'$\langle \hat{n}_i \rangle$')
plt.xticks(np.arange(0,Nsites,1))
plt.show()