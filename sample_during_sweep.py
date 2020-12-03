""" Routines for direct componentwise sampling during a DQMC sweep."""

import numpy as np

# from sample_Slater import sample_FF_GreensFunction
def sample_FF_GreensFunction(G, Nsamples, update_type='low-rank'):
    """
       Component-wise direct sampling of site occupations from a *spinless* free fermion
       pseudo density matrix in the grand-canonical ensemble.

       Parameters:
       ----------- 
            G: Free fermion Green's function G_ij = Tr( \\rho c_i c_j^{\\dagger} )
               for a fermionic pseudo density matrix \\rho. 
            Nsamples: Number of occupation number configurations to be generated
               from the given pseudo density matrix.
            update_type: 'naive' or 'low-rank'
               Update the correction due to inter-site correlations either by inverting 
               a matrix or by a more efficient low-rank update which avoids matrix 
               inversion altogether.
       Returns:
       --------
            A Fock state of occupation numbers sampled from the input free-fermion 
            pseudo density matrix.
            The Fock state carries a sign as well as a reweighting factor, which takes 
            care of the sign problem (type I), which is due to the fact that the pseudo density 
            matrix is non-Hermitian.
    """
    G = np.array(G, dtype=np.float64)
    assert(len(G.shape) == 2)
    assert(G.shape[0] == G.shape[1])

    # dimension of the single-particle Hilbert space
    D = G.shape[0]

    corr = np.zeros(D, dtype = np.float64)
    cond_prob = np.zeros(2, dtype = np.float64)

    for ss in np.arange(Nsamples):
        #print("isample_per_HS=%d"%(ss))
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
        
        Xinv = np.zeros((1,1), dtype=np.float64)
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
                # Avoid computation of determinants and inverses altogether
                # by utilizing the formulae for determinant and inverse of 
                # block matrices. 
                # Compute Xinv based on the previous Xinv. 
                g =  1.0/(G[k,k] - occ - corr[k])  #(-1)**occ * 1.0/cond_prob[occ]  # This latter expression also works. 
                uu = np.matmul(Xinv, G[Ksites, k])
                vv = np.matmul(G[k, Ksites], Xinv)
                Xinv_new = np.zeros((k+1,k+1), dtype=np.float64)
                Xinv_new[np.ix_(Ksites, Ksites)] = Xinv[np.ix_(Ksites, Ksites)] + g*np.outer(uu, vv)
                Xinv_new[k, Ksites] = -g*vv[Ksites]
                Xinv_new[Ksites, k] = -g*uu[Ksites]
                Xinv_new[k,k] = g
                Xinv = Xinv_new   

            Ksites = Ksites + list([k])

        assert(len(occ_vector) == D)

        yield np.array(occ_vector), sign, reweighting_factor


class Sampler( object ):
    """
        Structure with parameters for componentwise
        direct sampling during a DQMC sweep.
        
        Initializes output files. 
    """
    def __init__(self, Nsamples_per_HS, list_of_sitearrays, outfiles):
        Nspecies = 2
        assert( len(outfiles) == Nspecies )
        assert( len(list_of_sitearrays) == 1 ) # temporary restriction (only sample the full system since it is small anyway) => IMPROVE
        self.Nsamples_per_HS = Nsamples_per_HS
        self.list_of_sitearrays = list_of_sitearrays
        self.outfiles = outfiles
        
        # write header to outfiles (overwriting previous content) 
        fh = ['','']
        with open(self.outfiles[0], 'w') as fh[0], open(self.outfiles[1], 'w') as fh[1]:
            for s in np.arange(Nspecies):
                header  = "# The sampling of a pseudofermion density matrix results in a sign problem (type I)."
                header += "# Additionally there may be a sign problem arising from the BSS algorithm itself (type II)."
                header += "# BSS_sign | sampling sign (real part) | sampling sign (imag. part) "
                header += "| reweighting factor | occupation vector ... \n"
                fh[s].write(header)
    
    
def sample_during_sweep(G, Sparam):
    """
        Perform direct componentwise sampling at a given 
        imaginary time slice for which the equal-time Green's functions 
        G[0:Nspecies] are provided. 
        
        Precondition: 
            The Sampler object `Sparam` must have been initialized 
            with the desired number of samples per HS configuration
            and the names of the output files. 
    """
    from numpy import linalg 
    
    Nspecies = 2
    assert(len(G) == Nspecies)
    assert( all( G[i].shape == G[i+1].shape for i in np.arange(len(G)-1) ) )
    assert(isinstance(Sparam, Sampler))
    
    # If the Hamiltonian itself already exhibits a sign problem 
    # (sign problem of type II), then the Fock states generated in the 
    # sampling procedure need to be reweighted. 
    # 'BSS' stands for Blankenbecler-Scalapino-Sugar as in the BSS algorithm. 
    BSS_weight = 1.0
    for species in np.arange(Nspecies):
        BSS_weight *= linalg.det(np.eye(*G[species].shape) + G[species])
    BSS_sign = np.sign(BSS_weight)
    
    out_fh = ['','']
    with open(Sparam.outfiles[0], 'a') as out_fh[0], open(Sparam.outfiles[1], 'a') as out_fh[1]:            
        # Sample different subsystems A (or the whole system)
        for sitesA in Sparam.list_of_sitearrays:
            for species in np.arange(Nspecies):
                # generator object
                generate_Fock_states = sample_FF_GreensFunction(G=G[species][np.ix_(
                    sitesA, sitesA)], Nsamples=Sparam.Nsamples_per_HS, update_type='low-rank')

                batch = []
                for occ_vector, sign, reweighting_factor in generate_Fock_states:
                    strline  = "%f %f %f %f     " % (BSS_sign, sign.real, sign.imag, reweighting_factor)
                    strline += ' '.join( map(str, occ_vector) ) + '\n'
                    batch.append(strline)
                    
                out_fh[species].writelines(batch)

