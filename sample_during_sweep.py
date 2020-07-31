""" Routines for direct componentwise sampling during a DQMC sweep."""

import numpy as np

from sample_Slater import sample_FF_GreensFunction

class Sampler( object ):
    """
        Structure with parameters for componentwise
        direct sampling during a DQMC sweep.
        
        Initializes output files. 
    """
    def __init__(self, Hub, Nsamples_per_HS, list_of_sitearrays, outfiles):
        assert( isinstance(Hub, Hubbard) )
        assert( len(outfiles) == Hub.Nspecies )
        assert( len(list_of_sitearrays) == 1 ) # temporary restriction (only sample the full system since it is small anyway) => IMPROVE
        self.Nsamples_per_HS = Nsamples_per_HS
        self.list_of_sitearrays = list_of_sitearrays
        self.outfiles = outfiles
        
        # write header to outfiles (overwriting previous content) 
        with open(self.outfiles[0], 'w') as fh[0], open(self.outfiles[1], 'w') as fh[1]:
            for s in np.arange(Hub.Nspecies):
                header  = "# The sampling of a pseudofermion density matrix results in a sign problem (type I).")
                header += "# Additionally there may be a sign problem arising from the BSS algorithm itself (type II).")                
                header += "# BSS_sign | sampling sign (real part) | sampling sign (imag. part) "
                header += "| reweighting factor | occupation vector ... \n"
                fh[s].write(header)
    
    
def sample_during_sweep(Hub, G, Sparam):
    """
        Perform direct componentwise sampling at a given 
        imaginary time slice for which the equal-time Green's functions 
        G[0:Nspecies] are provided. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(len(G) == Hub.Nspecies)
    assert( all( G[i].shape == G[i+1].shape for i in np.arange(len(G)-1) ) )
    assert(isinstance(Sparam, Sampler))
    
        # If the Hamiltonian itself already exhibits a sign problem 
        # (sign problem of type II), then the Fock states generated in the 
        # sampling procedure need to be reweighted. 
        # 'BSS' stands for Blankenbecler-Scalapino-Sugar as in the BSS algorithm. 
        BSS_weight = 1.0
        for species in np.arange(Hub.Nspecies):
            BSS_weight *= linalg.det(np.eye(*G[species].shape) + G[species])
        BSS_sign = np.sign(BSS_weight)
        
        out_fh = ['','']
        with open(Sparam.outfiles[0], 'a') as out_fh[0], open(Sparam.outfiles[1], 'a') as out_fh[1]:            
            # Sample different subsystems A (or the whole system)
            for sitesA in Sparam.list_of_sitearrays:
                for species in np.arange(Hub.Nspecies):
                    # generator object
                    generate_Fock_states = sample_FF_GreensFunction(G=G[species][np.ix_(
                        sitesA, sitesA)], Nsamples=Sparam.Nsamples_per_HS, update_type='low-rank')

                    batch = []
                    for occ_vector, sign, reweighting_factor in generate_Fock_states:
                        strline  = "%f %f %f %f     " % (BSS_sign, sign.real, sign.imag, reweighting_factor)
                        strline += ' '.join( map(str, occ_vector) ) + '\n'
                        batch.append(strline)
                        
                    out_fh[species].writelines(batch)

