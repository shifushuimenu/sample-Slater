#!/usr/bin/python3.5

import numpy as np
from scipy import linalg, allclose
from profilehooks import profile

from sample_Slater import sample_SlaterDeterminant, prob2cumul, bisection_search
from test_suite import prepare_test_system_finiteT

# REMOVE
import matplotlib.pyplot as plt 
# REMOVE

# REMOVE
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

# REMOVE    


def sample_FF_pseudo_density_matrix(OBDM, Nsamples):
    """
        PROTOTYPE: NOT ALL OPERATIONS ARE EFFICIENTLY IMPLEMENTED 
        (NOT WORKING CORRECTLY FOR PSEUDO DENSITY MATRICES, 
        gives the same results as sample_FF_OBDM())

        Sample Fock states from a free fermion pseudo density matrix \rho,
        which arises naturally in determinantal QMC simulations. 

        \rho is non-hermitian and does not admit a spectral decomposition. 

        The sampling of \rho proceeds in two steps. After calculating the distribution of 
        particle numbers p[N] ("full counting statistics") of \rho, each direct sampling 
        step consists of 
          1. sampling a particle number sector of \rho according to p[N] => Np particles
          2. sampling the sector of fixed particle number Np analogously to the way 
             a single Slater determinant is sampled 
            (see Yuan Wan's notes about free fermion pseudo density matrices).

        Input: 
            OBDM: Equal-time one-body density matrix.
            Nsamples

        Return type:
            Generator object which outputs Nsamples occupation number states. 

    """

    assert( OBDM.shape[0] == OBDM.shape[1] ) 
    Nsites = OBDM.shape[0] # number of sites = number of orbitals = max. total number of fermions 

    # Calculate "full counting statistics"
    p_N = particle_number_distribution(OBDM=OBDM)
    # We will need the principal minors of the following matrix: 
    expmX = linalg.inv( np.eye(Nsites) - OBDM ) - np.eye(Nsites)
    # In the case of non-Hermitian X, we need the SVD of exp(-X).
    U, singular_values, Vh = linalg.svd( expmX )

    # partition sum 
    print(U)
    print(Vh)
    print("singular_values=", singular_values)
    print(linalg.det( np.eye(Nsites) + expmX ))
    print(linalg.det( np.matmul(U.transpose(), Vh.transpose()) + np.diag(singular_values) )*linalg.det(U)*linalg.det(Vh) )
    print( "sum(s[:])=", np.prod(singular_values) )
    exit()

    for ss in range(Nsamples):

        # 1. Sample particle number 
        Np = sample_particle_number_sectors(prob_N = p_N)

        # 2. Sample the selected particle number sector of \rho. 
        # Randomly pick an index set I of Np numbers from [0,1,...,Nsites].
        I = np.sort(np.random.permutation(np.arange(Nsites))[0:Np])
        nu_rndvec = np.random.permutation(np.arange(Np))

        occ_vector = sample_SlaterDeterminant(U=expmX[:,I], nu_rndvec=nu_rndvec)
        print('sample nr. =%d'%(ss), occ_vector)        
        yield occ_vector



def sample_FF_OBDM(OBDM, Nsamples_per_mode_occ, Nsamples_per_Sdet):
    """
        Sample Fock states from a spinless free fermion (FF) one-body density matrix (OBDM).
        In general, the OBDM is not hermitian. 

        Return type:
            Generator object which outputs occupation number states. 

            As a figure of merit, the accumulated histogram of the natural orbital occupations 
            ("momentum distribution function") of the OBDM is also returned.

        Comment: 
            When the OBDM results from a Hubbard-Stratonovich sample in a determinantal QMC simulation,
            it is in general not a hermitian matrix since, in this case, the free-fermion system is coupled 
            to a time-dependent fluctuating field.
    """

    assert(len(OBDM.shape)==2 and (OBDM.shape[0]==OBDM.shape[1])), "Expect a square matrix as OBDM."
    Ns = OBDM.shape[0]
    # v[:,i] is the right eigenvector corresponding to w[i]
    w, v = linalg.eig(OBDM)
    # At (moderately) low temperature, the eigenvalues of the OBDM form a sharp Fermi surface:
    # The mode occupations are either zero or one. 
    # Order in descending order according to occupation probabilities.
    ind = np.argsort(w)     # WRONG
    w = w[ind[::-1]]; v = v[:,ind[::-1]]  # WRONG

    assert( np.all(w.imag == 0) ), "OBDM has imaginary eigenvalues: sum(abs(w.imag)) = %12.8f" % sum(abs(w.imag))
    p_occ = w  

    # sample the mode occupancies
    norm = np.trace(OBDM)
    print("norm rho_A=", norm)

    sampled_already = set()
    sum_p_conf = 0.0

    hist_mode_occupation = np.zeros(Ns, dtype='float32')

    for jj in range(Nsamples_per_mode_occ):
        r = np.random.random(size=(Ns,))
        occ_vector_modes = np.array([1 if r[i] < p_occ[i] else 0 for i in range(Ns)], dtype='int32')
        prob_mode_occupation = np.prod(np.where(occ_vector_modes==1, w, 1-w))
        hist_mode_occupation += occ_vector_modes

        occ_hash = tuple(occ_vector_modes)
        if ( occ_hash not in sampled_already):
            sampled_already.add(occ_hash)
            # total probability so far of the generated configuration of occupatiopn numbers
            sum_p_conf = sum_p_conf + prob_mode_occupation            
        print("=============================================")
        print('occ_vector_modes=', occ_vector_modes)
        print('prob_mode_occupation=', prob_mode_occupation)
        print("=============================================")

        # Construct the Slater determinant.
        col_idx = np.where(occ_vector_modes==1)[0]
        Sdet = v[:, col_idx]

        # print("Single particle wave functions are only properly normalized if their complex-valuedness", \
        #       "is taken into account properly.")
        # print([sum(abs(Sdet[:,i])**2) for i in range(Sdet.shape[-1])])

        # Direct sampling: 
        # ================
        Np = sum(occ_vector_modes) # number of particles in the Slater determinant 

        for ii in range(Nsamples_per_Sdet):

            occ_vector = sample_SlaterDeterminant(U=Sdet, nu_rndvec=np.random.permutation(np.arange(Np)))
            print('sample nr. =%d'%(ii), occ_vector)        
            yield occ_vector

    
    print("========= CHECK ===================")
    print("hist_mode_occupations=", hist_mode_occupation / float(Nsamples_per_mode_occ))
    print("REAL(p_occ)=", p_occ.real)
    print("IMAG(p_occ)=", p_occ.imag)
    plt.plot(range(len(p_occ)), p_occ.real, '-o', label=r'$\langle n_{\alpha}\rangle$')
    plt.plot(range(len(p_occ)), hist_mode_occupation / float(Nsamples_per_mode_occ), '-x', label='histogram')
    plt.legend(loc="upper right")
    plt.xlabel(r'natural orbital index $\alpha$')
    plt.ylabel(r'$\langle n_{\alpha} \rangle$')
    plt.show()
    print("========= CHECK ===================")


def particle_number_distribution(OBDM):
    """
        Compute the full counting statistics of the particle number given a 
        free fermion one-body density matrix (OBDM).

        Input:
            OBDM: equal-time one-body density matrix of a free fermion system 
        
        Returns:
            prob_N: normalized particle number distribution function 
    """

    assert( OBDM.shape[0] == OBDM.shape[1] )
    Nsites = OBDM.shape[0]

    # 1. Calculate the generating function chiN(phi) from 
    # the eigenvalues of the OBDM.
    xi = linalg.eigvals(OBDM).real
    chi_N = np.zeros(Nsites+1, dtype=np.complex64)
    chi_N = [ np.prod(1.0 + (np.exp(1j*(2*np.pi*n / (Nsites + 1)))- 1.0)*xi[:]) for n in np.arange(0, Nsites+1)]
    
    # 2. The normalized particle number distribution is obtained from the discrete inverse Fourier 
    # transform of the generating function.
    prob_N_tmp = np.zeros(Nsites+1, dtype='complex64')
    for Np in np.arange(0, Nsites+1):
        prob_N_tmp[Np] = np.sum([ np.exp(-1j*(2*np.pi*n*Np / (Nsites + 1))) * chi_N[n] for n in np.arange(0, Nsites+1) ])
    prob_N_tmp[:] /= (Nsites+1)
    # Ceck whether the imaginary parts of the probability distribution have cancelled.        
    assert( allclose(prob_N_tmp.imag, 0.0) )
    prob_N = np.zeros(Nsites+1, dtype='float32')
    prob_N = prob_N_tmp.real

    return prob_N


def sample_particle_number_sectors(prob_N):
    """
        Sample a total particle number N from the probability 
        distribution of total particle numbers ("full counting statistics").
    """
    tol=1e-7
    assert(abs(1.0 - sum(prob_N)) < tol)

    cumul_prob = prob2cumul( prob_vec=prob_N )
    N = bisection_search( prob=np.random.rand(), cumul_prob_vec=cumul_prob )
    
    return N 


def principal_minor(A, index_set):
    """
        Compute the principal minor ("inclusive" definition)
            det( A_{i1,i2,...,iM; i1,i2,...,iM})
        of the square N x N matrix A for the ordered index set 
        i1 < i2 < i3 < ... < iM.
    """

    A = np.array(A); index_set = np.array(index_set)
    assert( A.shape[0] == A.shape[1] ); N = A.shape[0]
    assert( len(index_set.shape) == 1)
    assert( index_set.shape[0] <= A.shape[0] )
    assert( np.max(index_set) <= A.shape[0] )

    index_set = np.sort(index_set)
    return linalg.det(A[np.ix_(index_set, index_set)])


def all_principal_minors(A):
    """
        Return all principal minors of a square matrix A.
    """
    A = np.array(A)
    assert( A.shape[0] == A.shape[1] ); N = A.shape[0]

    for j in range(N):
        for i in range(j):
            for k in range(i):
                index_set = np.array([k,i,j])
                yield  principal_minor(A, index_set)


if __name__ == '__main__':

    # =============================
    #  OBDM from determinantal QMC
    # =============================
    OBDM_tot = np.loadtxt("../test_data/OBDM_L16x16_U7.2mu3.6_attractiveHubbard.dat")
    # OBDM = OBDM_tot[np.ix_(np.arange(Ns//2,3*Ns//4), np.arange(Ns//2,3*Ns//4))]
    #OBDM = OBDM_tot[np.ix_([102,103],[102,103])]
    #np.savetxt("questionable_OBDM.dat", OBDM)
    OBDM = square_region(OBDM_tot, L_A=4, x0=5, y0=5)
    assert( OBDM.shape[0] == OBDM.shape[1] )
    Nsites = OBDM.shape[0]
    beta=8.0
    mu=3.6

    # p_N = particle_number_distribution(OBDM=OBDM)
    # plt.plot(np.arange(len(p_N)), p_N)
    # plt.show()

    # # Are all principal minors of exp(-X) positive ? 
    # # =================================================================
    # expmX = linalg.inv( np.eye(Nsites) - OBDM ) - np.eye(Nsites)

    # for minor_val in all_principal_minors(expmX):
    #     print(minor_val)


#  #   ## =======================================
#  #   ##  Thermal OBDM for free fermion system
#  #   ## ========================================
#  #   Nsites, beta, mu, MDF, OBDM = prepare_test_system_finiteT(Nsites=81, beta=5.00, mu=0.0, potential='random-binary')

#     NsModes=10  # How often are the natural orbitals of the many-body density matrix sampled to construct a Slater determinant ? 
#     NsDet=10     # How often is each Slater determinant sampled ? 
  
# #    # generator function 
# #    generate_Fock_states = sample_FF_OBDM(OBDM, Nsamples_per_mode_occ=NsModes, Nsamples_per_Sdet=NsDet)

    generate_Fock_states = sample_FF_pseudo_density_matrix(OBDM=OBDM, Nsamples=1000)
 
    av_density = np.zeros(Nsites)
    av2_density = np.zeros(Nsites)

    ss = 0
    for occ_vector in generate_Fock_states:
        ss += 1
        #print('sample nr. =%d'%(ss), occ_vector)
        av_density += occ_vector
        av2_density += occ_vector**2
    Nsamples = ss
    av_density /= float(Nsamples)
    av2_density /= float(Nsamples)
    sigma_density = np.sqrt(av2_density - av_density**2) / np.sqrt(Nsamples)
    print("sum_av_density=", sum(av_density))
    print("sum(np.diag(OBDM))=", sum(np.diag(OBDM)))
    print(sigma_density)     
    
# #    # plot
#     plt.errorbar(np.arange(Nsites), av_density, yerr=sigma_density, fmt='-o', \
#             label='sampling Slater determinant \n Nsamples = %d' % (Nsamples) \
#                    + ' \n NsModes = %d' % (NsModes))
#     plt.plot(np.arange(Nsites), np.diag(OBDM), label='from Green\'s function')
#     plt.xlabel('site i')
# #    # plt.title('Free fermions in a parabolic trap.\n'+r'Nsites=%d, $\beta$=%4.2f, $\mu$=%4.2f' %(Nsites,beta,mu))
# #    plt.title('Free fermions in a random binary potential.\n'+r'Nsites=%d, $\beta$=%4.2f, $\mu$=%4.2f' %(Nsites,beta,mu))
#     plt.ylabel(r'$\langle \hat{n}_i \rangle$')
#     plt.legend(loc='lower center')
# #    plt.show()
# #    plt.savefig('FF_trap_sample_FF_OBDM_L%d_beta%4.2f_NperModes%d_NperDet%d.png' \
# #               % (Nsites, beta, NsModes, NsDet))

#     # =============================
#     #  OBDM from determinantal QMC
#     # =============================   
#     plt.title('HS sample from DQMC simulations.\n'+r'Nsites=%d, $\beta$=%4.2f, $\mu$=%4.2f' %(Nsites,beta,mu))            
#     plt.savefig('HSsample_U-7.2mu3.6beta8_L%d_OBDM_LA4_NperModes%d_NperDet%d.png' \
#                % (Nsites, NsModes, NsDet))               
#     plt.show()
