#!/usr/bin/env python3
"""
    Read in a sequence of equal-time single-particle  Green\'s functions
    output from the ALF DQMC code.
"""

import numpy as np

from ast import literal_eval as make_tuple
import re 
from io import StringIO

__all__ = ['read_GreenF', 'read_GreenF_spinful']

def F2PY_cmplx(string):
    """
        Converter function: 
        Convert a complex number in Fortran output format
        to a numpy complex number.

        Example:
              string = '(1.0, 1.0)'  ->  1 + 1j
    """
    #t = make_tuple(string.decode('utf-8').strip())
    t = make_tuple(string.strip())
    return np.complex(t[0], t[1])


def read_GreenF(file_object, dtype=np.complex64):
   """
       Read the equal-time Green's function, which is a complex 
       matrix, from a file with complex numbers written in Fortran format,
       i.e. (1.0,1.0) -> 1+1j

       Green's functions for different Hubbard-Stratonovich (HS) samples 
       are stored in the same file, with different instances separated 
       by two empty lines. 

       Keep reading Green's functions till the end of the file is reached.

       Input: 
       ======
          file_object: an open file handle
          dtype: np.complex64 or np.float64
       
       Returns:
       ========
          A generator object yielding a complex matrix. 
          If dtype=np.float64, the imaginary parts of the matrix elements
          are discarded. 
   """

   rows = []
   while True:
       line = file_object.readline()

       if not line:
           print("End of file reached.")
           break

       # Remove leading and trailing whitespaces and replace 
       # a sequence of succesive whitespaces by a single whitespace.
       fields = re.sub('\s+', ' ', line.strip()).split(' ')

       if (fields==['']):
           # skip the second separating empty line so that the 
           # next read starts from the first data line.
           line = file_object.readline()
           if not line:
               print("End of file reached.")
               break
           # yield the Green's function 
           if (not (rows == [])):
               G = np.vstack(tuple(rows))
               rows=[]
               assert(G.shape[0] == G.shape[1]), \
                 'Green\'s function must be square matrix. G.shape= (%d, %d)' % (G.shape[0], G.shape[1])
               yield np.array(G, dtype=dtype)
       else:
           rows.append([F2PY_cmplx(c) for c in fields])


def read_GreenF_spinful(file_object_tuple, dtype=np.float64):
   """
       Read the equal-time Green's function, which is a complex 
       matrix, from a file with complex numbers written in Fortran format,
       i.e. (1.0,1.0) -> 1+1j

       Green's functions for different Hubbard-Stratonovich (HS) samples 
       are stored in the same file, with different instances separated 
       by two empty lines. 

       Keep reading Green's functions till the end of the file is reached.

       For a spinful model, such as the Hubbard model, the Green's functions 
       for the two spin species in the same HS configuration need to be read 
       in simultaneously. Green's functions for different species are stored 
       in different files. 

       Input: 
       ======
          file_object_tuple: a tuple of open file handles of length 'Nspecies' 
          dtype: np.complex64 or np.float64
       
       Returns:
       ========
          A generator object yielding an array of complex matrices of length 'Nspecies'. 
          If dtype=np.float64, the imaginary parts of the matrix elements
          are discarded. 

       Note: Each input file containing Green's functions must be terminated 
             by two empty lines. 
   """

   assert(dtype in (np.complex64, np.float64)), 'Unknown data type in input file.'

   Nspecies = len(file_object_tuple)
   file_objects = tuple(file_object_tuple)

   rows = []
   G_tot = []

   EOF_counter = 0

   while (EOF_counter < Nspecies): 
        for species in np.arange(Nspecies):
            while True:
                line = file_objects[species].readline()

                if not line:
                    print("EOF reached; species=%d" % species)
                    EOF_counter += 1
                    break

                # Remove leading and trailing whitespaces and replace 
                # a sequence of succesive whitespaces by a single whitespace.
                fields = re.sub('\s+', ' ', line.strip()).split(' ')

                if (fields==['']):
                    # Skip the second separating empty line so that the 
                    # next read starts from the first data line.
                    line = file_objects[species].readline()

                    if not line:
                        print("Badly formatted input file."
                              "Each input Green's function should be terminated by"
                              "two empty lines; species=%d" % species)
                        break

                    if (not (rows == [])):
                        G = np.vstack(tuple(rows))
                        rows=[]
                        assert(G.shape[0] == G.shape[1]), \
                            'Green\'s function must be square matrix. G.shape= (%d, %d)' % (G.shape[0], G.shape[1])
                        ##  print('appending G, species=%d' % species)                
                        if (species == (Nspecies-1)):
                            G_tot.append(G)
                            # Yield the Green's functions for a given HS sample as a stack
                            # of matrices for different spin species. 
                            G_out = np.array(G_tot, dtype=dtype).copy()
                            G_tot = []
                            ## print("yielding G")
                            yield G_out 
                            break
                        else:
                            G_tot.append(G)
                            break                               
                else:
                    if (dtype == np.complex64):
                        rows.append([F2PY_cmplx(c) for c in fields])
                    else:
                        rows.append([np.float(c) for c in fields])
    


if __name__ == '__main__':

#    with open('Green_dn_several.dat') as fh:
#        for counter, G in enumerate(read_GreenF(fh, dtype=np.float64)):
#            if (counter >= 4):
#                break
#            print(G)

#    with open('Green_up_short.dat') as fh_up:
#        with open('Green_dn_short.dat') as fh_dn:
#            for G in read_GreenF_spinful((fh_up, fh_dn), dtype=np.complex64):
#                print('G.shape=', G.shape)
#                print("================")
    

   with open('Green_up_ncpu000.dat') as fh_up:
        with open('Green_dn_ncpu000.dat') as fh_dn:
            for G in read_GreenF_spinful((fh_up, fh_dn)):
                print('G.shape=', G.shape)
                print("================")
 

# TODO: - add docstrings with unit tests    
#       - comments to make the control flow easier to understand
#       - cleaner logic 