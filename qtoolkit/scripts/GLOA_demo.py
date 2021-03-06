# ======================================================================
# Copyright CERFACS (October 2018)
# Contributor: Adrien Suau (adrien.suau@cerfacs.fr)
#
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

import numpy

import qtoolkit.algorithms.group_leader as gloa
import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.constants.operations as qopconsts
import qtoolkit.utils.timeit as qtimeit


def gatesequence2str(basis, basis_str, gate_sequence):
    def gate2str(gate, gate_str, param) -> str:
        return gate_str + (f"({param})" if callable(gate) else "")

    sequence = gate_sequence.gates
    parameters = gate_sequence.params
    return "".join([gate2str(basis[i], basis_str[i], parameters[i]) for i in sequence])


# Define the parameters of the algorithm
basis = [
    qopconsts.H,
    qopconsts.T,
    qopconsts.T.H,
    qopconsts.Rx,
    qopconsts.Ry,
    qopconsts.Rz,
]
bounds = [
    None,
    None,
    None,
    numpy.array([[0], [2 * numpy.pi]]),
    numpy.array([[0], [2 * numpy.pi]]),
    numpy.array([[0], [2 * numpy.pi]]),
]
timer = qtimeit.Timer()

# 1. Generate the random unitary we want to approximate.
# U = sud_gen.generate_random_SUd(2)
U = numpy.array(
    [
        [0.113_266_73 + 0.649_633_26j, -0.396_781_88 + 0.638_522_84j],
        [0.396_781_88 + 0.638_522_84j, 0.113_266_73 - 0.649_633_26j],
    ]
)

# 3. Approximate it.
timer.tic()
cost, U_approx = gloa.group_leader(
    U, length=40, n=10, p=15, basis=basis, max_iter=20, parameters_bound=bounds
)
timer.toc("GLOA algorithm")

print("Decomposition characteristics:")
print(f"Fowler error: {qdists.fowler_distance(U, U_approx.matrix)}.")
print(f"Trace error:  {qdists.trace_distance(U, U_approx.matrix)}.")
print(f"GLOA cost:    {cost}.")
print(f"Gate count:   {len(list(U_approx.operations))}")
