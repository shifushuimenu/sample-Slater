#!/usr/bin/python3.5
"""
    Sample Fock states from a stream of Green's functions for different Hubbard-Stratonovich 
    field configurations, for two spin species. 
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import resource
import os 

from mpi4py import MPI

from sample_Slater import sample_FF_GreensFunction

from test_suite import occ2int_spinless, occ2int_spinful
from read_GreenF import read_GreenF_spinful

# MPI communicator
comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()

# check that mpi works correctly
print("rank %5.5d of %5.5d" % (MPI_rank, MPI_size))


Nsites = 4
list_of_sitearrays = ([0, 1, 3, 2],)
assert(isinstance(list_of_sitearrays, tuple))
assert(np.all([len(a)==Nsites for a in list_of_sitearrays]))
max_HS_samples = 5
Nsamples_per_HS = 50
skip = 0

N_spin_species = 2
dimH1 = dimH2 = 2**Nsites
dimH = dimH1*dimH2
prob_Fock_states = np.zeros(dimH, dtype=np.float64)
prob_Fock_states2 = np.zeros(dimH, dtype=np.float64)

Green_infile = ('GreenF_ncpu%5.5d_up.dat' % (MPI_rank),
                'GreenF_ncpu%5.5d_dn.dat' % (MPI_rank))
# Check that files exist.                               
if not np.all([os.path.isfile(f) for f in Green_infile]):
    print("Green's function files not found.")
    exit()

ss = 0
ss_HS = 0

Fock_states_updn = np.zeros(
    (N_spin_species, Nsamples_per_HS, Nsites), dtype=np.int8)
weight_updn = np.zeros((N_spin_species, Nsamples_per_HS), dtype=np.float64)
sign_updn = np.zeros((N_spin_species, Nsamples_per_HS), dtype=np.int8)

with open(Green_infile[0]) as fh_up:
    with open(Green_infile[1]) as fh_dn:
        for counter, G in enumerate(read_GreenF_spinful((fh_up, fh_dn), dtype=np.float64)):
            if (counter < skip):
                continue
            if (counter >= max_HS_samples):
                break

            ss_HS += 1
            # print("ss_HS=", ss_HS)

            # Sample different subsystems A
            # ([0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]):
            for sitesA in list_of_sitearrays:

                Fock_states_updn[...] = 0
                weight_updn[...] = 0.0
                sign_updn[...] = 0

                for species in np.arange(N_spin_species):
                    # generator object
                    generate_Fock_states = sample_FF_GreensFunction(G=G[species][np.ix_(
                        sitesA, sitesA)], Nsamples=Nsamples_per_HS, update_type='low-rank')

                    sss = 0
                    for occ_vector, sign, weight in generate_Fock_states:

                        Fock_states_updn[species, sss, :] = occ_vector[:]
                        weight_updn[species, sss] = weight
                        sign_updn[species, sss] = sign
                        sss += 1
                        # print("sss=", sss)

                # combine the Fock states of the different spin species
                for i in np.arange(Nsamples_per_HS):
                    occ_vector_up = Fock_states_updn[0, i, :]
                    occ_vector_dn = Fock_states_updn[1, i, :]
                    weight = weight_updn[0, i] * weight_updn[1, i]
                    sign = sign_updn[0, i] * sign_updn[1, i]

                    idx = occ2int_spinful(occ_vector_up, occ_vector_dn)
                    prob_Fock_states[idx] += sign * weight
                    prob_Fock_states2[idx] += weight**2
                    ss += 1


Nsamples = ss
N_HS_samples = ss_HS
prob_Fock_states /= float(Nsamples)
prob_Fock_states2 /= float(Nsamples)

# Here, the autocorrelation time sbould be properly computed !
sigma = np.sqrt(prob_Fock_states2 - prob_Fock_states**2) / np.sqrt(ss_HS)


# Every MPI process writes out its results
# idx of prob_Fock_states[idx] is specified by occ2int_spinful()
MM = np.vstack((np.arange(dimH), prob_Fock_states[:], sigma[:])).transpose()
column_labelling = "# state index  |  probability   |  standard deviation \
(over different MCMC Green's function samples in one Markov chain)"
header = '# direct componentwise sampling: max_HS_samples = %d, Nsamples_per_HS = %d' \
    % (max_HS_samples, Nsamples_per_HS)
header_labels = header + "\n" + column_labelling
np.savetxt('prob_Fock_ncpu%5.5d.dat' % (MPI_rank), MM, header=header_labels)

# ===================================================
# MPI reduce: Average over independent Markov chains.
# ===================================================
send_prob = prob_Fock_states.copy()
send_prob2 = prob_Fock_states.copy()**2

if MPI_rank == 0:
    recv_prob = np.zeros(send_prob.shape)
    recv_prob2 = np.zeros(send_prob2.shape)
    comm.Reduce(send_prob, recv_prob, op=MPI.SUM, root=0)
    comm.Reduce(send_prob2, recv_prob2, op=MPI.SUM, root=0)
else:
    comm.Reduce(send_prob, None, op=MPI.SUM, root=0)
    comm.Reduce(send_prob2, None, op=MPI.SUM, root=0)

if MPI_rank == 0:
    prob_Fock_allCPUs = recv_prob / float(MPI_size)
    prob2 = recv_prob2 / float(MPI_size)
    prob_squared = prob_Fock_allCPUs**2
    # Central Limit Theorem for independent Markov chains.
    sigma_allCPUs = np.sqrt(prob2 - prob_squared) / np.sqrt(float(MPI_size))

    # Root process writes out the average and standard deviation over independent Markov chains.
    MM_allCPUs = np.vstack(
        (np.arange(dimH), prob_Fock_allCPUs[:], sigma_allCPUs[:])).transpose()
    header = header + "\nNumber of independent Markov chains = %d" % (MPI_size)
    header_labels = header + "\n" + column_labelling
    np.savetxt('prob_Fock_allCPUs.dat', MM_allCPUs, header=header_labels)