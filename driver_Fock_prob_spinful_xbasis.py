#!/usr/bin/python3.5
"""
    Sample a stream of Green's functions for different Hubbard-Stratonovich 
    field configurations, for one spin species.
"""

import numpy as np
import matplotlib.pyplot as plt 
import time 
import resource

from sample_Slater import sample_FF_GreensFunction, rotate_GreensFunction

from test_suite import occ2int_spinless, occ2int_spinful
from read_GreenF import read_GreenF_spinful

N_spin_species = 2
Nsites = 8
dimH1 = dimH2 = 2**Nsites
dimH = dimH1*dimH2
prob_Fock_states = np.zeros(dimH, dtype=np.float32)
prob_Fock_states2 = np.zeros(dimH, dtype=np.float32)

Green_infile = ('Green_up.dat', 'Green_dn.dat')

max_HS_samples=4000
skip = 0
ss=0
ss_HS=0

Nsamples_per_HS = 10
Fock_states_updn = np.zeros((Nsamples_per_HS, 2*Nsites), dtype = np.int8)
weight_updn = np.zeros((Nsamples_per_HS), dtype = np.float32)
sign_updn = np.zeros((Nsamples_per_HS), dtype = np.int8)

with open(Green_infile[0]) as fh_up:
    with open(Green_infile[1]) as fh_dn:
        for counter, G in enumerate(read_GreenF_spinful((fh_up, fh_dn), dtype=np.float32)):
            if (counter < skip):
                continue
            if (counter >= max_HS_samples):
                break

            ss_HS += 1
            print("ss_HS=", ss_HS)

            # subsystem A
            for sitesA in ([0,1,2,3,4,5,6,7],): # [0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]):

                Fock_states_updn[...] = 0
                weight_updn[...] = 0.0
                sign_updn[...] = 0

                G_rotated = rotate_GreensFunction(G_up=G[0][np.ix_(sitesA, sitesA)], G_dn=G[1][np.ix_(sitesA, sitesA)], quant_axis='y')

                generate_Fock_states = sample_FF_GreensFunction(G=G_rotated, Nsamples=Nsamples_per_HS, update_type='naive')

                sss = 0
                for occ_vector, sign, weight in generate_Fock_states:                    

                    Fock_states_updn[sss, :] = occ_vector[:]
                    weight_updn[sss] = weight
                    sign_updn[sss] = sign
                    sss += 1
                        # print("sss=", sss)

                # combine the Fock states of the different spin species 
                for i in np.arange(Nsamples_per_HS):            
                    occ_vector_up = Fock_states_updn[i, 0:Nsites]
                    occ_vector_dn = Fock_states_updn[i, Nsites:2*Nsites]
                    weight = weight_updn[i]
                    sign = sign_updn[i]

                    print("sign*weight=", sign*weight)

                    idx = occ2int_spinful(occ_vector_up, occ_vector_dn)
                    prob_Fock_states[idx] += sign * weight
                    prob_Fock_states2[idx] += weight**2
                    ss += 1                


Nsamples = ss
N_HS_samples = ss_HS
prob_Fock_states /= float(Nsamples)
prob_Fock_states2 /= float(Nsamples)

# Plot 
# Here, the autocorrelation time sbould be properly computed !
sigma = np.sqrt(prob_Fock_states2 - prob_Fock_states**2) / np.sqrt(Nsamples)

np.savetxt('prob.dat', prob_Fock_states, header='# batch of %d HS samples' % max_HS_samples)

print("sum(prob_Fock_states)=", sum(prob_Fock_states), '+/-', sum(sigma))
fig, ax1 = plt.subplots(1,1)
ax1.set_title('. \n Nsamples per HS = %d, N_HS samples = %d' % (Nsamples_per_HS, N_HS_samples))
ax1.errorbar(np.arange(dimH), prob_Fock_states, yerr=sigma, fmt='+b', \
        label='probability')      
prob_Fock_ED = np.loadtxt('prob_Fock.dat') #('prob_Fock_ED_U2_beta1_halffilled_2x2.dat')         
#ax1.plot(prob_Fock_ED[:,0], prob_Fock_ED[:,1], '-r', label='Exact diagonalization')
#ax1.legend(loc='lower center')
plt.xlabel(r'index $i$')
plt.ylabel(r'$P(i)$')
#plt.xticks(np.arange(0,Nsites,1))
plt.show()