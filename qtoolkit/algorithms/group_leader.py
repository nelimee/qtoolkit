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

"""Implementation of the Group Leader Optimisation Algorithm.

This modules implements the GLOA algorithm as presented in
https://arxiv.org/abs/1004.2242.
"""

import typing

import numpy

import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.gloa.gate_sequence_population import \
    GateSequencePopulation
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence


def group_leader(unitary: qtypes.UnitaryMatrix, length: int, p: int, n: int,
                 basis: typing.Sequence[qtypes.SUdMatrixGenerator],
                 parameters_bound: typing.Optional[
                     numpy.ndarray] = None) -> QuantumGateSequence:
    """Implementation of the Group Leader Optimisation Algorithm.

    See https://arxiv.org/abs/1004.2242 for details about the algorithm.

    :param unitary: A unitary matrix representing a quantum gate (or
    circuit) to decompose.
    :param basis: The basis used for the decomposition.
    :param length: Length of the generated sequences.
    :param parameters_bound: Bounds for the parameters of the basis gates.
    :return:
    """
    r = numpy.array([0.8, 0.1, 0.1])
    correctness_weight = 1
    circuit_cost_weight = 0
    circuit_cost_func = lambda x: 1

    population = GateSequencePopulation(basis, unitary, length, n, p, r,
                                        correctness_weight, circuit_cost_weight,
                                        circuit_cost_func, parameters_bound)
    for i in range(100):
        population.perform_mutation_and_recombination()
        population.perform_one_way_crossover()

    return population.get_leader()
