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

"""Implement the :py:class:`~.QuantumCircuitPopulation` class used in GLOA.

The :py:class:`~.QuantumCircuitPopulation` stores the groups of
:py:class:`~.QuantumCircuit`.
"""

import copy
import typing

import numpy

import qtoolkit.data_structures.gloa.quantum_circuit_group as gsg
import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop
import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.types as qtypes


class QuantumCircuitPopulation:
    """A list of several :py:class:`QuantumCircuitGroup` instances."""

    def __init__(self, basis: typing.Sequence[qop.QuantumOperation],
                 objective_unitary: qtypes.UnitaryMatrix, length: int, n: int,
                 population: int, r: numpy.ndarray, correctness_weight: float,
                 circuit_cost_weight: float,
                 circuit_cost_func: qcirc.CircuitCostFunction,
                 parameters_bounds: qtypes.Bounds = None) -> None:
        """Initialise the :py:class:`~.QuantumCircuitPopulation` instance.

        :param basis: a sequence of allowed operations. The operations can be
            "abstract" (i.e. with None entries, see the documentation for the
            :py:class:`~.QuantumOperation` class) or not (i.e. with specified
            entries).
        :param objective_unitary: unitary matrix we are trying to approximate.
        :param length: length of the sequences that will be generated.
        :param n: number of groups.
        :param population: population of the group, i.e. number of gate
            sequences contained in this group.
        :param r: rates determining the portion of old (r[0]), leader (r[1]) and
            random (r[2]) that are used to generate new candidates.
        :param correctness_weight: scalar representing the importance attached
            to the correctness of the generated circuit.
        :param circuit_cost_weight: scalar representing the importance attached
            to the cost of the generated circuit.
        :param circuit_cost_func: a function that takes as input an instance of
            :py:class:`~.QuantumCircuit` and returns a float representing the
            cost of the given circuit.
        :param parameters_bounds: a list of bounds for each operation in the
            `basis`. A None value in this list means that the corresponding
            operation is not parametrised. A None value for the whole list
            (default value) means that no gate in `basis` is parametrised.
        """
        self._groups = [
            gsg.QuantumCircuitGroup(basis, objective_unitary, length,
                                    population, r, correctness_weight,
                                    circuit_cost_weight, circuit_cost_func,
                                    parameters_bounds) for _ in range(n)]
        self._population = population
        self._length = length
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func
        self._objective_unitary = objective_unitary

    def perform_one_way_crossover(self) -> None:
        """Apply the one way crossover step of the GLOA.

        See the `GLOA paper <https://arxiv.org/abs/1004.2242>`_ for more
        precision on this step.
        """
        # One parameter per gate, length gates for each sequences, p sequences.
        number_of_parameters = self._length * self._population
        for group in self._groups:
            # Number of cross-over we will perform on the current group.
            crossover_number = numpy.random.randint(
                number_of_parameters // 2 + 1)
            # Perform the cross-overs.
            for _ in range(crossover_number):
                # Pick at random the indices.
                circuit_index = numpy.random.randint(self._population)
                operation_index = numpy.random.randint(self._length)
                # Create a new quantum gate sequence from the old one.
                new_circuit = copy.copy(group.circuits[circuit_index])
                current_params = new_circuit[operation_index].parameters
                if current_params is not None:
                    new_params = self._get_parameters_from_group(group, len(
                        current_params))
                    new_circuit[operation_index].parameters = new_params
                old_cost = group.costs[circuit_index]
                new_cost = qdists.gloa_objective_function(new_circuit,
                                                          self._objective_unitary,
                                                          self._correctness_weight,
                                                          self._circuit_cost_weight,
                                                          self._circuit_cost_func)
                if new_cost < old_cost:
                    group.circuits[circuit_index] = new_circuit
                    group.costs[circuit_index] = new_cost
        return

    @staticmethod
    def _get_parameters_from_group(group: gsg.QuantumCircuitGroup,
                                   parameter_number: int) -> typing.Sequence[
        float]:
        circuits_ids = list(range(len(group.circuits)))
        parameters = []
        while len(parameters) < parameter_number and circuits_ids:
            poped_circ_id = numpy.random.randint(len(circuits_ids))
            circ = group.circuits[circuits_ids.pop(poped_circ_id)]
            operations_ids = list(range(len(circ.size)))
            while len(parameters) < parameter_number and operations_ids:
                poped_op_id = numpy.random.randint(len(operations_ids))
                operation = circ[poped_op_id]
                if operation.parameters:
                    parameters = parameters + list(operation.parameters)
        return parameters[:parameter_number]

    def perform_mutation_and_recombination(self) -> None:
        """Perform the mutation and recombination step on each group."""
        for group in self._groups:
            group.mutate_and_recombine()

    def get_leader(self) -> typing.Tuple[float, qcirc.QuantumCircuit]:
        """Get the global leader.

        The global leader is the leader of the group formed by the leaders of
        each stored groups.

        :return: the global leader and its cost.
        """
        costs_and_leaders = [group.get_leader() for group in self._groups]

        leader_costs = [cost_and_leader[0] for cost_and_leader in
                        costs_and_leaders]

        best_cost_idx: int = numpy.argmin(leader_costs)

        return costs_and_leaders[best_cost_idx]
