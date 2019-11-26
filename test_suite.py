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


def prepare_test_system_finiteT(Nsites=21, beta=1.0, mu=0.0):
    """
        One-dimensional system of free fermions with Nsites sites
        in an external trapping potential. 

        Input:
            beta = inverse temperature 
            mu = chemical potential 
        Output:
            Return the one-body density matrix (OBDM)
                <c_i^{\dagger} c_j> = Tr(e^{-beta H}c_i^{\dagger} c_j)
            and the occupations of natural orbitals (momentum-distribution
            function for a translationally-invariant system) as a vector.
    """
    assert(Nsites%2==1), "test_suites: Nsites should be an odd number."

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

    sp_energies, U = linalg.eigh(H)

    # fugacity
    z = np.exp(beta*mu)
    # momentum distribution function (for a translationally-invariant system)
    # or natural orbital occupancies (for a trapped system)
    MDF = np.diag(z*np.exp(-beta*sp_energies) / (1 + z*np.exp(-beta*sp_energies)))
    OBDM = np.matmul(np.matmul(U, MDF), U.conj().T)

    return (Nsites, beta, mu, np.sort(np.diag(MDF))[::-1], OBDM)