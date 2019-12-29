#!/usr/bin/python3.5
"""
    Sample a stream of Green's functions for different Hubbard-Stratonovich 
    field configurations, for one spin species.
"""

import numpy as np
import matplotlib.pyplot as plt 
import time 
import resource

from sample_Slater import sample_FF_GreensFunction

from test_suite import occ2int_spinless, occ2int_spinful
from read_GreenF import read_GreenF


Nsites = 4
dimH = 2**4
prob_Fock_states = np.zeros(dimH, dtype=np.float32)
prob_Fock_states2 = np.zeros(dimH, dtype=np.float32)

max_HS_samples=20
ss=0
ss_HS=0
with open('Green_dn_several.dat') as fh:
    for counter, G in enumerate(read_GreenF(fh, dtype=np.float32)):
        if (counter >= max_HS_samples):
            break
        print(G)
        ss_HS += 1
        # generator object
        generate_Fock_states = sample_FF_GreensFunction(G=G, Nsamples=1000, update_type='low-rank')

        for occ_vector, sign, weight in generate_Fock_states:
            ss += 1
            idx = occ2int_spinless(occ_vector)
            prob_Fock_states[idx] += sign * weight
            prob_Fock_states2[idx] += weight**2

Nsamples = ss
N_HS_samples = ss_HS
prob_Fock_states /= float(Nsamples)
prob_Fock_states2 /= float(Nsamples)

# Plot 
# Here, the autocorrelation time sbould be properly computed !
sigma = np.sqrt(prob_Fock_states2 - prob_Fock_states**2) / np.sqrt(Nsamples)

print(sum(prob_Fock_states), '+/-', sum(sigma))
fig, ax1 = plt.subplots(1,1)
ax1.set_title('. \n Nsamples per HS = %d, N_HS samples = %d' % (Nsamples, N_HS_samples))
ax1.errorbar(np.arange(dimH), prob_Fock_states, yerr=sigma, fmt='-+', \
        label='average density')       
#ax1.plot(np.arange(dimH), prob_Fock_states, label='blabla')
ax1.legend(loc='lower center')
plt.xlabel(r'index $i$')
plt.ylabel(r'$P(i)$')
#plt.xticks(np.arange(0,Nsites,1))
plt.show()