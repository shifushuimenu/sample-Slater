#!/usr/bin/python3.5

import numpy as np
from scipy import linalg, allclose
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
            [1] arXiv:1806.00472

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
    # n-th particle (The constant factor 1/(n!) in Ref. [1] can be neglected for the 
    # unnormalized probability distribution.)
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
            for l,j in  enumerate(col_idx):
                for k,i in enumerate(row_idx_sample):
                    Amat[k,l] = U[i,j]
            cond_prob[x_sample] = abs(linalg.det(Amat))**2

        cumul_prob, norm = prob2cumul(cond_prob)
        x = bisection_search( prob=np.random.rand(), cumul_prob_vec=cumul_prob )
        occ_vec[x] = 1
        row_idx = row_idx + list([x])

    return occ_vec


def sample_nonorthogonal_SlaterDeterminant(U, singular_values, Vh, nu_rndvec):
    """
        THIS DOES NOT WORK !
        Input:
            After singular value decomposition 
                \exp(-X) = U @ S @ Vh
            U:  Unitary matrix having left singular values as columns (Slater determinant no. 1)
            singular_values: 
            Vh: Unitary matrix having right singular vectors as rows (Slater determinant no. 2).  
            nu_rndvec: A random permutation of the integers [1,2,...,N]
        Output:
            A Fock state of occupation numbers sampled from the input
            Slater determinants. The Fock state is represented as a vector of
            0s and 1s.
    """
    U = np.array(U, dtype=np.complex64)
    Vh = np.array(Vh, dtype=np.complex64)
    singular_values = np.array(singular_values, dtype=np.float32)
    nu_rndvec = np.array(nu_rndvec, dtype=np.int32)

    print("U.shape=", U.shape)
    print("Vh.shape=", Vh.shape)
    print("singular_values=", singular_values)

    assert( U.shape[0] == Vh.shape[1] and U.shape[1] == Vh.shape[0] ), \
        "shape mismatch: U: (%d, %d), Vh: (%d, %d)" % (U.shape + Vh.shape)

    # M is the number of orbitals, N is the total number of particles 
    (M,N) = U.shape
    assert ( nu_rndvec.size == N ), "nu_rndvec.size = %d, M = %d" % (nu_rndvec.size, N)
    assert ( singular_values.size == N )

    # Occupation numbers of the M orbitals:
    # occ_vec[i]==1 for occupied and occ_vec[i]==0 for unoccupied i-th orbital.
    occ_vec = np.zeros(M, dtype=np.int8)

    # Sample orbitals for N particles iteratively.
    row_idx = []; col_idx = []
    # unnormalized conditional probability cond_prob(x) for choosing orbital x for the
    # n-th particle (The constant factor 1/(n!) in Ref. [1] can be neglected for the 
    # unnormalized probability distribution.)
    cond_prob = np.zeros(M, dtype=np.float32)
    for n in range(N):
        cond_prob[...] = 0.0
        # collect row and column indices
        col_idx = col_idx + list([nu_rndvec[n]])
        for x_sample in range(M):
            row_idx_sample = row_idx + list([x_sample])
            assert (len(col_idx) == len(row_idx_sample))
            # construct submatrix
            Amat = U[np.ix_(row_idx_sample, col_idx)]
            Bmat = Vh[np.ix_(col_idx, row_idx_sample)]
            # product of singular values 
            zz = np.prod(singular_values[col_idx])
            xx = abs(linalg.det(Amat)*linalg.det(Bmat)) # avoid neg. prob. of the form -0.00000
            # print("zz=", zz, "zz*xx=", zz*xx.real)
            assert( xx.imag == 0 ), 'Product of determinants has imaginary part.'
            cond_prob[x_sample] = zz * xx.real

        # if (all(cond_prob < 1e-8)):
        #     occ_vec[:] = 0
        #     break

        cumul_prob, norm = prob2cumul(cond_prob)
        x = bisection_search( prob=np.random.rand(), cumul_prob_vec=cumul_prob )
        occ_vec[x] = 1
        row_idx = row_idx + list([x])

    return occ_vec


def sample_FF_GreensFunction(G, Nsamples, update_type='low-rank'):
    """
       Component-wise sampling of site occupations from a free fermion
       pseudo density matrix in the grand-canonical ensemble.

       Input: 
            G: Free fermion Green's function G_ij = Tr( \rho c_i c_j^{\dagger} )
               for a fermionic pseudo density matrix \rho. 
            Nsamples: Number of occupation number configurations to be generated
            update_type: 'naive' or 'low-rank'
               Update the correction due to inter-site correlations either by inverting 
               a matrix or by a more efficient low-rank update which avoids matrix 
               inversion altogether.
       Output:
            A Fock state of occupation numbers sampled from the input free-fermion 
            pseudo density matrix.
            The Fock state carries a sign as well as a reweighting factor, which takes 
            care of the sign problem.
    """
    G = np.array(G, dtype=np.float32)
    assert(len(G.shape) == 2)
    assert(G.shape[0] == G.shape[1])

    # dimension of the single-particle Hilbert space
    D = G.shape[0]

    corr = np.zeros(D, dtype = np.float32)
    cond_prob = np.zeros(2, dtype = np.float32)

    for ss in np.arange(Nsamples):
        corr[...] = 0.0
        sign = 1.0
        reweighting_factor = 1.0
        # Component-wise direct sampling (k=0)
        k=0
        Ksites = [k]
        cond_prob[1] = 1.0 - G[k,k]
        cond_prob[0] = G[k,k]
        if (np.random.random() < cond_prob[1]):
            occ_vector = [1]
        else:
            occ_vector = [0]
        occ = occ_vector[0]
        Xinv = np.zeros((1,1), dtype=np.float32)
        Xinv[0,0] = 1.0/(G[0,0] - occ)
        # Component-wise direct sampling  (k=1,...,D-1)
        for k in np.arange(1,D):
            # "Correction" due to correlations between sites
            if (update_type == 'naive'):
                corr[k] = np.matmul(G[k, Ksites], np.matmul(linalg.inv(G[np.ix_(Ksites,Ksites)] - np.diag(occ_vector)), G[Ksites, k]))
            elif (update_type == 'low-rank'):       
                corr[k] = np.matmul( G[k, Ksites], np.matmul( Xinv, G[Ksites, k] ) )
            else:
                sys.exit('Error: Unkown update type')

            cond_prob[1] = 1 - G[k,k] + corr[k]
            cond_prob[0] = G[k,k] - corr[k]
            # Take care of quasi-probability distribution 
            if ((cond_prob[1] < 0) or (cond_prob[1] > 1)):                
                if (cond_prob[1] < 0):
                    norm = 1.0 - 2.0*cond_prob[1]
                    P = abs(cond_prob[1]) / norm
                else:
                    norm = 2.0*cond_prob[1] - 1.0 
                    P = cond_prob[1] / norm
                reweighting_factor *= norm
            else:
                P = cond_prob[1]
            if (np.random.random() < P):
                occ = 1
                sign *= np.sign(cond_prob[1])
            else:
                occ = 0
                sign *= np.sign(cond_prob[0])

            occ_vector = occ_vector + list([occ])   

            if (update_type == 'low-rank'):
                # Avoid computation of determiants and inverses altogether
                # by utilizing the formulae for determinant and inverse of 
                # block matrices. 
                # Compute Xinv based on the previous Xinv. 
                g =  1.0/(G[k,k] - occ - corr[k])  #(-1)**occ * 1.0/cond_prob[occ]  # This latter expression also works. 
                uu = np.matmul(Xinv, G[Ksites, k])
                vv = np.matmul(G[k, Ksites], Xinv)
                Xinv_new = np.zeros((k+1,k+1), dtype=np.float32)
                Xinv_new[np.ix_(Ksites, Ksites)] = Xinv[np.ix_(Ksites, Ksites)] + g*np.outer(uu, vv)
                Xinv_new[k, Ksites] = -g*vv[Ksites]
                Xinv_new[Ksites, k] = -g*uu[Ksites]
                Xinv_new[k,k] = g
                Xinv = Xinv_new   

            Ksites = Ksites + list([k])

        assert(len(occ_vector) == D)

        yield np.array(occ_vector), sign, reweighting_factor


def sample_expmX(U, N):
    """
        Input:
            U = exp(-X), an M-by-M matrix representing the free fermion pseudo density matrix. 
                exp(-X) is not necessarily hermitian.
                The principal minors of exp(-X) are the marginal probabilities we need. 
                They can be negative. 
            N: Number of particles in the given particle number sector: N \in {0,1,...,D}.
        Output:
            A Fock state of occupation numbers sampled from the input free-fermion pseudo density matrix.
            The Fock state carries a sign as well as a reweighting factor.
    """
    U = np.array(U, dtype=np.complex64)
    assert(U.shape[0] == U.shape[1])
    M = U.shape[0]
    assert(N<=M)

    # Occupation numbers of the M sites.
    # occ_vec[i]==1 for occupied and occ_vec[i]==0 for unoccupied i-th site.
    occ_vec = np.zeros(M, dtype=np.int8)
    # If a conditional probability is negative, shift the sign to the observable,
    # i.e. to the Fock state. 
    sign_vec = np.zeros(M, dtype=np.int8)

    # Sample sites for N particles iteratively.
    row_idx = []; col_idx = []
    # The conditional probability cond_prob(x) for choosing site x for the
    # n-th particle is the ratio of two principal minors of exp(-X).
    # Since this conditional probability can be negative for a pseudo density matrix,
    # the absolute value is taken. 
    cond_prob = np.zeros(M, dtype=np.float32)
    cond_prob_signed = np.zeros(M, dtype=np.float32)
    sign_structure = np.zeros(M, dtype=np.int8)
    row_idx = []
    reweighting_factor = 1.0 # reweighting factor due to the sign problem 
    sign = 1
    # Sample the positions of N particles. 
    for nn in range(N):
        cond_prob[...] = 0.0
        sign_structure[...] = 0.0
        # Collect row and column indices, which for a principal minor are the same,
        # to obtain all elements of the conditional probability.
        for x_sample in range(M):
            row_idx_sample = row_idx + list([x_sample])
            if (len(row_idx) >= 1):
                # The conditional probability is given as the ratio of principal minors.

                # Method 1: Low-rank update for iterative computation of the matrix inverse. 
                # Use the block determinant formula, which allows to avoid the computation 
                # of determinants altogether.
                                                

                # Method 2: Inverse of submatrix. Singular matrix for subsystem sizes larger than or equal to 5x5. 
                # Add invertibility noise to make the matrix non-singular. 
                v1 = U[np.ix_(list([x_sample]), row_idx)]
                v2 = U[np.ix_(row_idx, list([x_sample]))]
                dim = len(row_idx)
                invertibility_noise = (1e-9)*np.random.random((dim,dim))
                xx = U[x_sample, x_sample] - np.matmul(v1, np.matmul( linalg.inv(U[np.ix_(row_idx, row_idx)] + invertibility_noise), v2))

                # Method 3: Ratio of two determinants. Leads to numerical overflow of the extremely large determinants.
                #xx = linalg.det(U[np.ix_(row_idx_sample, row_idx_sample)]) / linalg.det(U[np.ix_(row_idx, row_idx)])
            else:
                # Determinant of a 1x1 matrix. 
                xx = U[np.ix_(row_idx_sample, row_idx_sample)]    
                # assert(xx > 0), "sign problem."
            assert( xx.imag == 0 ), 'Something is wrong with the determinant. (%15.10f, %15.10f)' % (xx.real, xx.imag)
            cond_prob_signed[x_sample] = xx.real
            cond_prob[x_sample] = abs(xx.real)
            sign_structure[x_sample] = np.sign(xx.real)

        cond_prob_signed, norm_signed = normalize(cond_prob_signed)
        cumul_prob, norm_unsigned = prob2cumul(cond_prob)
        x = bisection_search( prob=np.random.rand(), cumul_prob_vec=cumul_prob )
        #print(nn, cond_prob_signed, sum(cond_prob_signed[np.where(cond_prob_signed > 0)]), sum(cond_prob_signed[np.where(cond_prob_signed < 0)]), x)        
        occ_vec[x] = 1
        # taking care of the sign problem
        sign_vec[x] = sign_structure[x]
        sign *= sign_structure[x]
        reweighting_factor *= norm_signed / norm_unsigned

        row_idx = row_idx + list([x])
        col_idx = row_idx

    return occ_vec, sign_vec, sign, reweighting_factor


def normalize( quasi_prob ):
    """
        Normalize an array of numbers such that their sum is one.
        A (small) fraction of the numbers may be negative,
        thus representing a quasi-probability distribution. 

        Return:
            - normalized distribution
            - its normalization constant
    """

    quasi_prob = np.array(quasi_prob)
    norm = sum(quasi_prob)
    #assert(norm > 0), 'Quasi-probability distribution has predominantly negative values: norm=%15.8f.' % (norm)
    return quasi_prob[:] / norm, norm


def prob2cumul( prob_vec ):
    """
        For a vector of unnormalized probabilities, return a vector
        of cumulative probabilities and the normalization.
    """
    # REMOVE
    assert( not any(np.isnan(prob_vec)) ), print("prob_vec=", prob_vec)
    # REMOVE
    cumul = np.zeros(prob_vec.size)
    ss = 0.0
    for i in range(prob_vec.size):
        ss += prob_vec[i]
        cumul[i] = ss

    # Make sure that negative values below machine precision are set to zero.
    cumul[(cumul < 0) & (abs(cumul) < np.finfo('float32').eps)] = 0.0

    return cumul / ss, ss


def bisection_search( prob, cumul_prob_vec ):
    """
        Find the index idx such that
            cumul_prob(idx-1) < prob <= cumul_prob(idx).

        The indices into cumul_prob[:] start with zero.
    """
    cumul_prob_vec = np.array( cumul_prob_vec )

    # TEST
    for i, p in enumerate(cumul_prob_vec):
        if (p < 0):
            cumul_prob_vec[i] *= -1
    # TEST

    assert( all( cumul_prob_vec >= -np.finfo(float).eps ) ), print("cumul_prob_vec=", cumul_prob_vec)
    assert( cumul_prob_vec[-1] == 1.0 ), "cumul_prob_vec[-1]=%15.10f" % cumul_prob_vec[-1]

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
