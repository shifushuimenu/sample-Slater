#!/usr/bin/python3.5
# TODO:
#  - move particles only to neighbouring sites
#    and do not swap particles
#  - implement all sparse matrix-matrix multiplications
#  - store positions of the particles and carry an occupation vector
#    which is updated together with the Slater determinant
#
"""
    Markov chain Monte Carlo sampling of a Slater determinant.
"""

import numpy as np
import numpy.random
from scipy import linalg
from profilehooks import profile

from sample_Slater import unit, Slater2spOBDM

class OneBodyDensityMatrix:
    """
        Structure containing the current one-body density matrix (OBDM)
        < alpha | c_i^{+} c_j | psi >  (i,j = 0,...,Ns-1) for Fock state alpha.

        We separately store the matrices 'LRinv_timesL' and 'R' because only
        the first changes when updating the Fock state whereas the latter does
        not. 'LRinv_timesL' is updated by left-multiplication with a dense
        matrix and by right-multiplication with a sparse matrix.
    """
    def __init__(self, Ns, Np):
        # the actual one-body density matrix
        self.matrix = np.zeros((Ns,Ns), dtype=np.float64)
        # auxiliary matrices which are stored in memory for an efficient update
        # of the current OBDM.
        self.LRinv_timesL = np.zeros((Np, Ns), dtype=np.float64)
        self.R = np.zeros((Ns, Np), dtype=np.float64)

def random_Fock_state(Ns, Np):
    """
        Return the Slater determinant (rectangular matrix of dimension Ns x Np)
        which corresponds to a Fock state with Np fermions placed randomly in
        Ns orbitals.
    """
    assert (Ns >= Np)
    vec = np.hstack( (np.ones(Np), np.zeros(Ns-Np)) )
    # permute in place
    np.random.shuffle(vec)
    p=0
    for i in range(0, Ns):
        if (vec[i] == 1):
            if (p==0):
                alpha = unit(i, Ns)
            else:
                alpha = np.hstack((alpha, unit(i,Ns)))
            p=p+1
    return alpha


def Fock_to_occ_vector(Fock_state, occ_vector):
    """
        Input: Slater determinant representing a Fock state (occupation number
            state)
            Vector of occupation numbers (0s and 1s). Will be overwritten.
    """
    # IMPROVE: Store the occupation vector always together with the matrix
    # representing the Fock state as a Slater determinant so that this
    # function becomes redundant.
    (Ns, Np) = Fock_state.shape
    assert( occ_vector.shape == (Ns,) )
    occ_vector[:] = 0
    for i in range(Ns):
        for j in range(Np):
            if (Fock_state[i,j] == 1):
                occ_vector[i] = 1


def move(Fock_state, i, f):
    """
        Change the Fock state by moving a fermion from orbital i to
        orbital f.
        (IMPROVE: sparse matrix-matrix multiplication)
    """
    (Ns, Np) = Fock_state.shape
    P = np.eye(Ns)
    P[i, i] = P[f, f] = 0
    P[i, f] = P[f, i] = 1
    return np.matmul(P, Fock_state)

#    # # swap row i with row f
#    tmp = Fock_state[f,:]
#    Fock_state[f,:] = Fock_state[i,:]
#    Fock_state[i,:] = tmp
#    return Fock_state


def calc_Metropolis_ratio(OBDM, r, s):
    """
        Input:
            OBDM: Current "one-body density matrix" <alpha | c_j^{+} c_i | psi >
                between Slater determinant |psi> and Fock state |alpha>
                before the update.
            r: old position of the fermion in the current Fock state |alpha>
            s: new position of the fermion in the Fock state |alpha_new>
        Ouput:
            Metropolis update ratio for the transition from the current
            Fock state |alpha> to the new state |alpha_new> by moving a fermion
            from position r to s.
    """
    G = OBDM.matrix.transpose()
    # determinant ratio for the move (r -> s)
    ratio = (1.0 - (G[r,r] - G[s,r]))*(1.0 - (G[s,s] - G[r,s])) \
            - (G[r,r] - G[s,r])*(G[s,s] - G[r,s])
    return abs(ratio)**2


def make_spOBDM(alpha, psi):
    """
        Build the one-body density matrix
                OBDM_ij = < alpha | c_i^{+} c_j | psi >  (i,j = 0,...,Ns-1)
        from the Slater determinant | alpha > and | psi >.
        Input:
           |psi>: The Slater determinant wave function to be sampled,
                  represented by an Ns x Np rectangular matrix, where Ns is the
                  number of orbitals and Np is the number of fermions.
           |alpha>: Slater determinant of a Fock state, represented
                  by an Ns x Np rectangular matrix.
        Output:
           One-body density matrix.

    """
    assert (alpha.shape == psi.shape)
    (Ns, Np) = alpha.shape
    assert (Ns >= Np)
    # create OBDM object
    OBDM = OneBodyDensityMatrix(Ns, Np)
    L = np.array(alpha).transpose()
    R = np.array(psi)
    LRinv_timesL = np.matmul(linalg.inv(np.matmul(L,R)), L)

    OBDM.matrix = np.matmul(R, LRinv_timesL).transpose()
    OBDM.LRinv_timesL = LRinv_timesL
    OBDM.R = R
    return OBDM


def update_spOBDM(OBDM, r, s):
    """
        Update the OBDM after moving a fermion from site r to site s.
    """
    LL = OBDM.LRinv_timesL
    RR = OBDM.R
    (Ns, Np) = RR.shape # Ns = number of orbitals; Np = number of particles

    # correction matrices
    U = np.zeros((Np,4), dtype=np.float64)
    V = np.zeros((4,Np), dtype=np.float64)
    U[:,0] = LL[:,r]; U[:,1] = LL[:,s]; U[:,2] = U[:,0]; U[:,3] = U[:,1]
    V[0,:] = +RR[r,:]; V[1,:] = +RR[s,:]; V[2,:] = -RR[s,:]; V[3,:] = -RR[r,:]
    Delta = np.zeros((Ns,Ns), dtype=np.float64)
    Delta[r,r] = Delta[s,s] = 1.0; Delta[r,s] = Delta[s,r] = -1.0

    # Sherman-Morrison formula for updating R(LR)^{-1}L
    tmp1 = linalg.inv(np.eye(4) - np.matmul(V,U))
    tmp2 = np.matmul(np.matmul(U, tmp1), V)

    # dense matrix multiplication from the left
    C_dense = np.eye(Np) + tmp2
    tmp3 = np.matmul(C_dense, LL)
    # sparse matrix multiplication from the right
    # (IMPROVE  by avoiding full matrix multiplication)
    C_sparse = np.eye(Ns) - Delta
    OBDM.LRinv_timesL = np.matmul(tmp3, C_sparse)
    OBDM.matrix = np.matmul(RR, OBDM.LRinv_timesL).transpose()
    # OBDM.R does not change


#@profile
def sample_SlaterDeterminant_MCMC(psi, Nsweeps):
    """
        Input:
            | psi >: Rectangular Ns x Np matrix representing the Slater
            determinant to be sampled. Ns is the number of orbitals and Np
            the number of particles.
        Generates:
            Markov chain in the space of Fock states.
        Output:
            Occupation number vector after a fixed number of MC sweeps.
            Ideally, Nsweeps should be comparable to the autocorrelation time.
    """

    # Starting values
    (Ns, Np) = psi.shape
    alpha = random_Fock_state(Ns, Np)
    occ_vector = np.zeros(Ns, dtype=np.int8)
    Fock_to_occ_vector(alpha, occ_vector)
    OBDM = make_spOBDM(alpha, psi)
    # check particle number conservation
    ###assert (sum(np.diag(OBDM.matrix)) == Np)

    for i in range(Nsweeps):
        # One sweep consists in trying to move every particle to a neighbouring
        # unoccupied site.
        for pp in range(Np):
            # Attempt to move a fermion
            # IMPROVE: Make sure r is occupied and s is unoccupied, rather than choosing r and s randomly
            while True:
                (r, s) = np.random.randint(low=0, high=Ns, size=2)
                if (r != s): break
            # Metropolis accept / reject step
            R = calc_Metropolis_ratio(OBDM, r, s)
            ###print("R=%f" % R)
            if (np.random.rand() < R):
                # accept
                alpha = move(alpha, r, s)
                Fock_to_occ_vector(alpha, occ_vector)
                # print("r, s = %d, %d" % (r,s))
                # print(occ_vector)
                update_spOBDM(OBDM, r, s)
            else:
                pass
                # reject

    return occ_vector

def _test():
    import sample_Slater_MCMC, doctest
    return doctest.testmod(sample_Slater_MCMC)

if __name__ == '__main__':
    # run doctest unit test
    # _test()

    # free fermions in 1D (with open BC)
    # ==================================
    import matplotlib.pyplot as plt

    Nsites=101; Nparticles=50  # should be an odd number
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

    Nsamples = 10
    av_density = np.zeros(Nsites)
    av2_density = np.zeros(Nsites)
    for i in range(Nsamples):
        occ_vector = sample_SlaterDeterminant_MCMC(U[:,0:Nparticles], Nsweeps=20)
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
            label='MCMC sampling of Slater determinant \n Nsamples = %d' % (Nsamples))
    plt.plot(range(Nsites), density, label='from Green\'s function')
    plt.xlabel('site i')
    plt.title('Free fermions in a parabolic trap.\n Nparticlces = %d, Nsites=%d' %(Nparticles, Nsites))
    plt.ylabel(r'$\langle \hat{n}_i \rangle$')
    plt.legend(loc='lower center')
    plt.show()
