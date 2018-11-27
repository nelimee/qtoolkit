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

This modules implements a modified version of the GLOA algorithm
presented in https://arxiv.org/abs/1004.2242.

The changes made to the algorithm were mainly done to restrict the
algorithm to a specific set of operations, called "basis" in the code.
This modification may not be profitable to the algorithm: the prior idea
of its inventors was to use a very large set of allowed gates and then
rephrase these gates with sequences of gates in the basis.
A simpler version of the algorithm may be useful.
TODO: implement the real algorithm to test accuracy and speed.
"""

import typing

import numpy

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop
import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.gloa.quantum_circuit_population import (
    QuantumCircuitPopulation,
)


def group_leader(
    unitary: qtypes.UnitaryMatrix,
    length: int,
    basis: typing.Sequence[qop.QuantumOperation],
    n: int = 15,
    p: int = 25,
    parameters_bound: qtypes.Bounds = None,
    max_iter: int = 100,
    r: numpy.ndarray = None,
    correctness_weight: float = 0.9,
    circuit_cost_weight: float = 0.1,
    circuit_cost_func: qcirc.CircuitCostFunction = None,
) -> typing.Tuple[float, qcirc.QuantumCircuit]:
    """Implementation of the Group Leader Optimisation algorithm.

    The article presenting this algorithm can be found at
    https://arxiv.org/abs/1004.2242.

    :param unitary: unitary matrix we want to approximate.
    :param length: length of the resulting approximation sequence. The generated
        circuit will probably be simplifiable to a shorter quantum circuit, this
        parameter represents the length of the randomly generated quantum
        circuits, not the length of the simplified ones.
    :param basis: a sequence of allowed operations. The operations can be
        "abstract" (i.e. with None entries, see the documentation for the
        :py:class:`~.QuantumOperation` class) or not (i.e. with specified
        entries).
    :param n: number of groups used. See the article for a more complete
        description of the concept of "group".
    :param p: number of quantum circuit in each group.
    :param parameters_bound: bounds for the parameter(s) of the quantum gates in
        the basis. Bounds are specified as a sequence of 2-D numpy arrays. The
        first index (the index of the sequence) represent the gate concerned,
        i.e. parameters_bound[i] represents the bounds for basis[i]. For each
        gate, a bound is a 2-D numpy array. parameters_bound[i][0] is a 1-D
        numpy array with [parameter_number] floating-point values representing
        the lower-bounds of the parameters. Conversely, parameters_bound[i][1]
        is a 1-D numpy array with [parameter_number] floating-point values
        representing the upper-bounds of the parameters.
        If None, this means that no quantum gate in the basis is parametrised.
        If not all the quantum gates in the basis are parametrised, the
        parameter bounds corresponding to non-parametrised quantum gates can
        take any value (for example None).
    :param max_iter: maximum number of iteration performed by the algorithm.
    :param r: rates determining the portion of old (r[0]), leader (r[1]) and
        random (r[2]) that are used to generate new candidates. If None, the
        default value of the GLOA article is used: r = [0.8, 0.1, 0.1].
    :param correctness_weight: scalar representing the importance attached to
        the correctness of the generated circuit.
    :param circuit_cost_weight: scalar representing the importance attached to
        the cost of the generated circuit.
    :param circuit_cost_func: a function that takes as input an instance of
        QuantumGateSequence and returns a float representing the cost of the
        given circuit. If None, only the correctness will count and the circuit
        cost will be ignored.
    :return: the best QuantumGateSequence found to approximate the given unitary
        matrix.
    """
    if r is None:
        r = numpy.array([0.8, 0.1, 0.1])
    if circuit_cost_func is None:
        circuit_cost_weight = 0.0
        correctness_weight = 1.0

        def circuit_cost_func(_: qcirc.QuantumCircuit) -> float:
            return 1.0

    population = QuantumCircuitPopulation(
        basis,
        unitary,
        length,
        n,
        p,
        r,
        correctness_weight,
        circuit_cost_weight,
        circuit_cost_func,
        parameters_bound,
    )
    for i in range(max_iter):
        print(f"Iteration n°{i}/{max_iter}...", end=" ")
        population.perform_mutation_and_recombination()
        population.perform_one_way_crossover()
        print(f"[{population.get_leader()[0]}]")
    return population.get_leader()
