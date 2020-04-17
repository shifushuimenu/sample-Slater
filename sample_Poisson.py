# -*- coding: utf-8 -*-
import numpy as np

def sample_Poisson(G, Nsamples):
    """
        Sample a free fermion pseudo density matrix for one spin component 
                        *à la Poisson*, 
        i.e. by independently sampling each site with 
        a probability given by the corresponding diagonal element of the 
        Green's function. 
        
        Poisson sampling works correctly due to a symmetry in the HS field configurations 
        which is, apparently, present in the repulsive Hubbard model at half filling (and under 
        the condition that the HS decoupling is done in the spin channel)

       Input:
            G: Free fermion Green's function G_ij = Tr( \\rho c_i c_j^{\\dagger} )
               for a fermionic pseudo density matrix \\rho. 
            Nsamples: Number of occupation number configurations to be generated
       Output:
            A Fock state of occupation numbers
    """
    G = np.array(G, dtype=np.float64)
    assert(len(G.shape) == 2)
    assert(G.shape[0] == G.shape[1])

    # dimension of the single-particle Hilbert space
    D = G.shape[0]

    for ss in np.arange(Nsamples):
        # Independent sampling for each site.
        # Probability is given by the corresponding diagonal element of the Green's function. 
        occ_vector = []
        for k in np.arange(0,D):
            P = 1.0 - G[k,k]
            # assert(P>=0), "P=%15.10f"%P
            P = abs(P)
            if (np.random.random() < P):
                occ = 1
            else:
                occ = 0

            occ_vector = occ_vector + list([occ])   

        assert(len(occ_vector) == D)

        # for compatibility with the correct sampling routine 
        sign = 1.0
        weight = 1.0

        yield np.array(occ_vector), sign, weight