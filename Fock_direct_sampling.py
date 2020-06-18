#!/usr/bin/python3.5
"""
    Sample Fock states from a stream of Green's functions for different Hubbard-Stratonovich 
    field configurations, for two spin species, and produce a stream of Fock states.
    
    Input: Green's functions for spin up and down

    Output: - Fock states for spin up and spin down
            - the sign (type I) and reweighting factor of each Fock state
"""
import numpy as np
from numpy import linalg 
import time
import resource
import os 

from mpi4py import MPI
from MPI_parallel import * # make `MPI_rank` globally available
import argparse

from sample_Slater import sample_FF_GreensFunction

from test_suite import occ2int_spinless, occ2int_spinful
from read_GreenF import read_GreenF_spinful

# MPI communicator
comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()

parser = argparse.ArgumentParser(
    description="Sample Fock states ('snapshots') from fermionic pseudo-density matrices\
        as they arise naturally in finite-temperature determinantal QMC. The Green's functions\
        are assumed to be stored in the two synchronized files\
            `'Green_ncpu%5.5d_up.dat' % (MPI_rank)` and\
            `'Green_ncpu%5.5d_dn.dat' % (MPI_rank)`\
        with different HS samples separated by two empty lines and `MPI_rank` labelling\
        independent Markov chains.")
parser.add_argument("NsitesA", type=int, 
    help="Number of sites on subsystem A, where the sampling occurs. Set NsitesA=0 for sampling\
          all sites.")
parser.add_argument("list_of_sitearrays_file", metavar="filename", type=str,
    help="File with a list of sitearrays in the form `0 1 3 2\\n 8 9 16 17\\n...`\
         which specify the subsystems A (for exploiting translational invariance).\
         The length of each sitearray must match `NsitesA`.")    
parser.add_argument("--max_HS_samples", type=int,
    help="Maximum number of Green's functions to be read from file.")
parser.add_argument("--Nsamples_per_HS", type=int,
    help="Number of Fock states generated per Hubbard-Stratonovich (HS) field configuration.")
parser.add_argument("--skip", metavar="Nskip", type=int,
        help="Skip the first `Nskip` Green's functions entries from the file.")
args = parser.parse_args()


N_spin_species = 2
max_HS_samples = args.max_HS_samples
Nsamples_per_HS = args.Nsamples_per_HS
skip = args.skip

# check that mpi works correctly
print("rank %5.5d of %5.5d" % (MPI_rank, MPI_size))

# input file with Green's functions 
Green_infile = ('Green_ncpu%5.5d_up.dat' % (MPI_rank),
                'Green_ncpu%5.5d_dn.dat' % (MPI_rank))

# output file for sampled Fock states 
outfile = ('Fock_samples_ncpu%5.5d_up.dat' % (MPI_rank),
           'Fock_samples_ncpu%5.5d_dn.dat' % (MPI_rank))
for s in np.arange(N_spin_species):
    out_fh = open(outfile[s], 'w')
    out_fh.write("# The sampling of a pseudofermion density matrix results in a sign problem (type I).")
    out_fh.write("# Additionally there may be a sign problem arising from the BSS algorithm itself (type II).")
    out_fh.write("# The weight of a Hubbard-Stratonovich configuration (BSS weight) is not needed.")
    out_fh.write("# It is output anyways to check for numerical overflow, which would make the reported sign invalid.")
    out_fh.write("# BSS sign (type II)  |   BSS weight  |   sign (type I )  |   reweighting factor   |   occupation vector (one spin species only) \n#\n")
    out_fh.close()

# Check that the files exist.                               
if not np.all([os.path.isfile(f) for f in Green_infile]):
    print("Green's function files not found.")
    exit()

Nsites = args.NsitesA
if (Nsites != 0):
    A = np.loadtxt(args.list_of_sitearrays_file, dtype=int) #([0, 1, 3, 2],) # Note the last comma !
    list_of_sitearrays = tuple(l for l in A.reshape(-1, A.shape[-1]))
    assert(isinstance(list_of_sitearrays, tuple))
    assert(np.all([len(a)==Nsites for a in list_of_sitearrays]))
else:
    # get number of sites from the dimension of the Green's function
    with open(Green_infile[0]) as fh_up:
        Nsites = len(fh_up.readline().split())
    list_of_sitearrays = (np.arange(Nsites),)


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

            # If the Hamiltonian itself already exhibits a sign problem 
            # (sign problem of type II), then the Fock states generated in the 
            # sampling procedure need to be reweighted. 
            # 'BSS' stands for Blankenbecler-Scalapino-Sugar as in the BSS algorithm. 
            BSS_weight = 1.0
            for species in np.arange(N_spin_species):
                BSS_weight *= linalg.det(np.eye(*G[species].shape) + G[species])
            BSS_sign = np.sign(BSS_weight)               

            ss_HS += 1 

            out_fh = (open(outfile[0], 'a'), open(outfile[1], 'a'))
            # Sample different subsystems A (or the whole system)
            for sitesA in list_of_sitearrays:
                for species in np.arange(N_spin_species):
                    print("species=%d"%(species))
                    # generator object
                    generate_Fock_states = sample_FF_GreensFunction(G=G[species][np.ix_(
                        sitesA, sitesA)], Nsamples=Nsamples_per_HS, update_type='low-rank')

                    sss = 0
                    for occ_vector, sign, weight in generate_Fock_states:
                        sss += 1
                        print("sample Nr. %d" % sss)
                        strline = "%f %f %f %f     " % (BSS_sign, BSS_weight, sign, weight)
                        for b in occ_vector:
                            strline += str(b)+' '
                        out_fh[species].write(strline+'\n')

            out_fh[0].close()
            out_fh[1].close()                        
