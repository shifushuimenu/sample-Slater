#!/usr/bin/python3.5

import numpy as np
from scipy import linalg

def prepare_test_system_zeroT(Nsites=21):
    """
        One-dimensional system of free fermions with Nsites sites
        in an external trapping potential.
        Return the matrix of single-particle eigenstates. 
    """
    Nsites=21  # should be an odd number
    i0=int(Nsites/2)
    V = np.zeros(Nsites)
    t_hop = 1.0
    V_max = 1.0*t_hop   # max. value of the trapping potential at the edge of the trap
                        # (in units of the hopping)
    V_pot = V_max / i0**2
    for i in range(Nsites):
        V[i] = V_pot*(i-i0)**2

    H = np.zeros((Nsites,Nsites), dtype=np.float64)
    for i in range(Nsites):
        H[i,i] = V[i]
        if (i+1 < Nsites):
            H[i,i+1] = -t_hop
            H[i+1,i] = -t_hop

    eigvals, U = linalg.eigh(H)

    return (Nsites, U)


def prepare_test_system_finiteT(Nsites=21, beta=1.0, mu=0.0, potential='parabolic'):
    """
        One-dimensional system of free fermions with Nsites sites
        in an external trapping potential. 

        Input:
            beta = inverse temperature 
            mu = chemical potential 
            potential: Type of the external potenial which is either 'parabolic'
               or 'random-binary. 
        Output:
            Return the one-body density matrix (OBDM)
                <c_i^{\dagger} c_j> = Tr(e^{-beta H}c_i^{\dagger} c_j)
            and the occupations of natural orbitals (momentum-distribution
            function for a translationally-invariant system) as a vector.
    """
    #assert(Nsites%2==1), "test_suites: Nsites should be an odd number."

    i0=int(Nsites/2)
    V = np.zeros(Nsites)
    t_hop = 1.0

    if (potential == 'parabolic'):
        V_max = 1.0*t_hop   # max. value of the trapping potential at the edge of the trap
                        # (in units of the hopping)
        V_pot = V_max / i0**2
        for i in range(Nsites):
            V[i] = V_pot*(i-i0)**2
    elif (potential == 'random-binary'):
        absU = 7.2; dtau=0.05
        alphaU = np.arccosh(np.exp(dtau*absU/2.0))
        V = alphaU * np.random.random_integers(0,1,size=Nsites)
    else:
        print("Unknown type of external potential")
        exit()

    H = np.zeros((Nsites,Nsites), dtype=np.float64)
    for i in range(Nsites):
        H[i,i] = V[i]
        if (i+1 < Nsites):
            H[i,i+1] = -t_hop
            H[i+1,i] = -t_hop

    sp_energies, U = linalg.eigh(H)

    # fugacity
    z = np.exp(beta*mu)
    # momentum distribution function (for a translationally-invariant system)
    # or natural orbital occupancies (for a trapped system)
    MDF = np.diag(z*np.exp(-beta*sp_energies) / (1 + z*np.exp(-beta*sp_energies)))
    OBDM = np.matmul(np.matmul(U, MDF), U.conj().T)

    return (Nsites, beta, mu, np.sort(np.diag(MDF))[::-1], OBDM)


def square_region(OBDM, L_A, x0=1, y0=1):
    """
        Extract the elements of the OBDM which correspond to a square 
        region of the real-space lattice of size L_A x L_A.e 

        The lower left corner of the square region has coordinates (1,1).
    """
    Ns = OBDM.shape[0]
    L = int(np.sqrt(float(Ns)))
 
    row_idx = [(x0-1) + i + (y0-1)*L + (j-1)*L for j in range(1,L_A+1) for i in range(1,L_A+1)]
    col_idx = row_idx 

    return OBDM[np.ix_(row_idx, col_idx)]
