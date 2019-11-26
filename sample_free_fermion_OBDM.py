#!/usr/bin/python3.5

import numpy as np
from scipy import linalg
from profilehooks import profile

from sample_Slater import sample_SlaterDeterminant
from test_suite import prepare_test_system_finiteT

# REMOVE
import matplotlib.pyplot as plt 
# REMOVE

def create_random_OBDM(Ns=4):
    pass


def sample_FF_OBDM(OBDM, Nsamples_per_mode_occ, Nsamples_per_Sdet):
    """
        Sample Fock states from a spinless free fermion (FF) one-body density matrix (OBDM).
        In general, OBDM is not hermitian. 

        Return type:
            Generator object with outputs occupation number states. 

        Comment: 
            When the OBDM results from a HUbbard-Stratonovich sample in a determinantal QMC simulation,
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
    #ind = np.argsort(w)     # WRONG
    #w = w[ind[::-1]]; v = v[:,ind]  # WRONG

    p_occ = w  

    # sample the mode occupancies
    norm = np.trace(OBDM)
    print("norm rho_A=", norm)

    sampled_already = set()
    sum_p_conf = 0.0

    for jj in range(Nsamples_per_mode_occ):
        r = np.random.random(size=(Ns,))
        occ_vector_modes = np.array([1 if r[i] < p_occ[i] else 0 for i in range(Ns)], dtype='int32')
        prob_mode_occupation = np.prod(np.where(occ_vector_modes==1, w, 1-w))
        
        occ_hash = tuple(occ_vector_modes)
        if ( occ_hash not in sampled_already):
            sampled_already.add(occ_hash)
            # total probability so far of the generated configuration of occupatiopn numbers
            sum_p_conf = sum_p_conf + prob_mode_occupation            
        print("=============================================")
        print('occ_vector_modes=', occ_vector_modes)
        print('prob_mode_occupation=', prob_mode_occupation)
        print("=============================================")

        # construct Slater determinant
        col_idx = np.where(occ_vector_modes==1)[0]
        Sdet = v[:, col_idx]
        # print("Single particle wave functions are only properly normalized if their complex-valuedness", \
        #       "is taken into account properly.")
        # print([sum(abs(Sdet[:,i])**2) for i in range(Sdet.shape[-1])])
        # direct sampling: 
        # ================
        Np = sum(occ_vector_modes) # number of particles in the Slater determinant 
        nu_rndvecs = [ np.random.permutation(np.arange(Np)) for i in np.arange(Nsamples_per_Sdet) ]

        for ii in range(Nsamples_per_Sdet):

            occ_vector = sample_SlaterDeterminant(U=Sdet, nu_rndvec=nu_rndvecs[ii])
            # print('sample nr. =%d'%(i), occ_vector)        
            yield occ_vector


if __name__ == '__main__':

    # =============================
    #  OBDM from determinantal QMC
    # =============================
    # OBDM_tot = np.loadtxt("../test_data/OBDM_L16x16_U7.2mu3.6_attractiveHubbard.dat")
    # Ns = OBDM_tot.shape[0]
    # OBDM = OBDM_tot[np.ix_(np.arange(Ns//2,3*Ns//4), np.arange(Ns//2,3*Ns//4))]
    # print(OBDM.shape)
    # OBDM = OBDM_tot[np.ix_(np.arange(10,60),np.arange(10,60))]

     # =======================================
    #  Thermal OBDM for free fermion system
    # ========================================
    Nsites, beta, mu, MDF, OBDM = prepare_test_system_finiteT(Nsites=11, beta=2.50, mu=0.0)

    # generator function 
    NsModes=500   # How often are the eigenmodes of the many-body density matrix sampled ? 
    NsDet=2000   # How often is each Slater determinant sampled ? 
    generate_Fock_states = sample_FF_OBDM(OBDM, Nsamples_per_mode_occ=NsModes, Nsamples_per_Sdet=NsDet)

    av_density = np.zeros(Nsites)
    av2_density = np.zeros(Nsites)

    ss = 0
    for occ_vector in generate_Fock_states:
        ss += 1
        print('sample nr. =%d'%(ss), occ_vector)
        av_density += occ_vector
        av2_density += occ_vector**2
    Nsamples = ss
    av_density /= float(Nsamples)
    av2_density /= float(Nsamples)
    sigma_density = np.sqrt(av2_density - av_density**2) / np.sqrt(Nsamples)
    print("sum_av_density=", sum(av_density))
    print("sum(np.diag(OBDM))=", sum(np.diag(OBDM)))
    print(sigma_density)     
    
    # plot
    plt.errorbar(range(Nsites), av_density, yerr=sigma_density, fmt='-o', \
            label='sampling Slater determinant \n Nsamples = %d' % (Nsamples) \
                   + ' \n NsModes = %d' % (NsModes))
    plt.plot(range(Nsites), np.diag(OBDM), label='from Green\'s function')
    plt.xlabel('site i')
    plt.title(r'Free fermions in a parabolic trap.\\n Nsites=%d, $\beta$=%4.2f, $\mu$=%4.2f' %(Nsites,beta,mu))
    plt.ylabel(r'$\langle \hat{n}_i \rangle$')
    plt.legend(loc='lower center')
    plt.savefig('FF_trap_sample_FF_OBDM_L%d_beta%4.2f_NperModes%d_NperDet%d.png' \
               % (Nsites, beta, NsModes, NsDet))
    plt.show()