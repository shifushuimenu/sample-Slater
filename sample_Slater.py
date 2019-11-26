#!/usr/bin/python3.5

import numpy as np
from scipy import linalg
from profilehooks import profile

from test_suite import prepare_test_system_zeroT


def unit(i, M):
    """
        Create a unit column vector of matrix type
        with the i-th entry set to 1.
        M is the dimension of the vector space.
    """
    v = np.zeros(M)
    v[i] = 1.0
    return np.matrix(v).transpose()


def Slater2spOBDM(U):
    """
        Input:
           (M x N)-matrix U representing a Slater determinant with
           N particles in M orbitals.
        Output:
           Single-particle one-body density matrix (OBDM). The diagonal elements
           are the denstities in the M orbitals.
    """

    U = np.array(U)
    assert(len(U.shape) == 2); assert(U.shape[0] >= U.shape[1])
    (M,N) = U.shape

    GF = np.zeros((M,M)) # Green's function
    for i in range(M):
        P_prime1 = np.hstack((U, unit(i, M)))
        for j in range(M):
            P_prime2 = np.hstack((U, unit(j, M)))

            GF[i,j] = linalg.det(np.dot(np.transpose(P_prime1), P_prime2))

    # From Green's function to one-body density matrix
    OBDM = -GF
    for i in range(M):
        OBDM[i,i] += 1.0
    return OBDM

# @profile
def sample_SlaterDeterminant(U, nu_rndvec):
    """
        Step 2 of the algorithm for direct sampling of a Slater determinant.
        Reference:
            arXiv:1806.00472

        Input:
            U: Rectangular complex MxN matrix representing the Slater determinant
               of N spinless fermions. The columns of U are single-particle states.
            nu_rndvec: A random permutation of the integers [1,2,...,N]
        Output:
            A Fock state of occupation numbers sampled from the input
            Slater determinant. The Fock state is represented as a vector of
            0s and 1s.
    """
    U = np.array(U, dtype=np.complex64); nu_rndvec = np.array(nu_rndvec)
    (M,N) = U.shape
    assert ( nu_rndvec.size == N ), "nu_rndvec.size = %d, M = %d" % (nu_rndvec.size, N)

    # Occupation numbers of the M orbitals:
    # occ_vec[i]==1 for occupied and occ_vec[i]==0 for unoccupied i-th orbital.
    occ_vec = np.zeros(M, dtype=np.int8)

    # Sample orbitals for N particles iteratively.
    row_idx = []; col_idx = []
    # unnormalized conditional probability cond_prob(x) for choosing orbital x for the
    # n-th particle
    cond_prob = np.zeros(M)
    for n in range(N):
        cond_prob[...] = 0.0
        # collect row and column indices
        col_idx = col_idx + list([nu_rndvec[n]])
        for x_sample in range(M):
            row_idx_sample = row_idx + list([x_sample])
            assert (len(col_idx) == len(row_idx_sample))
            # construct submatrix
            Amat = np.zeros((len(col_idx), len(row_idx_sample)), dtype=np.complex64)
            for l,j in enumerate(col_idx):
                for k,i in enumerate(row_idx_sample):
                    Amat[k,l] = U[i,j]
            cond_prob[x_sample] = abs(linalg.det(Amat))**2

        cumul_prob = prob2cumul(cond_prob)
        x = bisection_search( prob=np.random.rand(), cumul_prob_vec=cumul_prob )
        occ_vec[x] = 1
        row_idx = row_idx + list([x])

    return occ_vec


def prob2cumul( prob_vec ):
    """
        For a vector of unnormalized probabilities, return a vector
        of cumulative probabilities.
    """
    # REMOVE
    assert( not any(np.isnan(prob_vec)) ), print("prob_vec=", prob_vec)
    # REMOVE
    cumul = np.zeros(prob_vec.size)
    ss = 0.0
    for i in range(prob_vec.size):
        ss += prob_vec[i]
        cumul[i] = ss
    return cumul / ss


def bisection_search( prob, cumul_prob_vec ):
    """
        Find the index idx such that
            cumul_prob(idx-1) < prob <= cumul_prob(idx).

        The indices into cumul_prob[:] start with zero.
    """
    cumul_prob_vec = np.array( cumul_prob_vec )
    assert( all(cumul_prob_vec >= 0) ), print("cumul_prob_vec=", cumul_prob_vec)
    assert( cumul_prob_vec[-1] == 1.0 )

    N = cumul_prob_vec.size

    FOUND = False; k1=0; k2=N-1
    while( not FOUND ):
        k = int((k1 + k2)/2.0)
        if (cumul_prob_vec[k] <= prob):
            k1 = k # k2 remains the same
            if ( (k2-k1) <= 1 ):
                FOUND = True
                idx = k2
        else:
            k2 = k # k1 remains the same
            if ( (k2-k1) <= 1 ):
                FOUND = True
                # The case prob < cumul_prob(k=0) is special because k=1 cannot be bracketed by k1 and k2.
                if ( prob < cumul_prob_vec[0] ):
                    idx = 0
                else:
                    idx = k2
    return idx


def _test():
    import sample_Slater, doctest
    return doctest.testmod(sample_Slater)
    

if __name__ == '__main__':
    # run doctest unit test
    # _test()

    # free fermions in 1D (with open BC)
    # ==================================
    import matplotlib.pyplot as plt

    Nsites, U = prepare_test_system_zeroT(Nsites=21)

    Nparticles=10; Nsamples=500
    eta_vec = np.arange(Nparticles)
    nu_rndvecs = [ np.random.permutation(eta_vec) for i in np.arange(Nsamples) ]

    av_density = np.zeros(Nsites)
    av2_density = np.zeros(Nsites)
    for i in range(Nsamples):
        occ_vector = sample_SlaterDeterminant(U[:,0:Nparticles], nu_rndvecs[i])
        print('sample nr. =%d'%(i), occ_vector)
        av_density += occ_vector
        av2_density += occ_vector**2
    av_density /= float(Nsamples)
    av2_density /= float(Nsamples)
    sigma_density = np.sqrt(av2_density - av_density**2) / np.sqrt(Nsamples)
    print(av_density)
    print(sigma_density)

    OBDM = Slater2spOBDM(U[:,0:Nparticles])
    density = np.diagonal(OBDM)
    print(density)

    # plot
    plt.errorbar(range(Nsites), av_density, yerr=sigma_density, fmt='-o', \
            label='sampling Slater determinant \n Nsamples = %d' % (Nsamples))
    plt.plot(range(Nsites), density, label='from Green\'s function')
    plt.xlabel('site i')
    plt.title('Free fermions in a parabolic trap.\n Nparticlces = %d, Nsites=%d' %(Nparticles, Nsites))
    plt.ylabel(r'$\langle \hat{n}_i \rangle$')
    plt.legend(loc='lower center')
    # plt.savefig('FF_trap_sampleSlaterDet_Np%d_Nsamples%d.png' % (Nparticles, Nsamples))
    plt.show()
