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

"""Implement the GateSequencePopulation class, used to implement GLOA."""

import typing

import numpy

import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.gloa.gate_sequence_group import GateSequenceGroup
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence


class GateSequencePopulation:
    """A data structure containing several GateSequenceGroup instances."""

    def __init__(self, basis: typing.Sequence[qtypes.SUdMatrixGenerator],
                 objective_unitary: qtypes.UnitaryMatrix, length: int, n: int,
                 p: int, r: numpy.ndarray, correctness_weight: float,
                 circuit_cost_weight: float, circuit_cost_func: typing.Callable[
            [QuantumGateSequence], float], parameters_bounds: typing.Optional[
            numpy.ndarray] = None) -> None:
        """Initialise the GateSequencePopulation instance.

        :param basis: gates available to construct the approximation. Each gate
        can be either a numpy.ndarray (which means that the gate is not
        parametrised) or a callable that takes a float as input and returns a
        numpy.ndarray representing the quantum gate.
        :param objective_unitary: unitary matrix we are trying to approximate.
        :param length: length of the sequences that will be generated in each
        group.
        :param n: number of groups.
        :param p: population of each group, i.e. number of gate sequences
        contained in each group.
        :param r: rates determining the portion of old (r[0]), leader (r[1]) and
        random (r[2]) that are used to generate new candidates.
        :param correctness_weight: scalar representing the importance attached
        to the correctness of the generated circuit.
        :param circuit_cost_weight: scalar representing the importance attached
        to the cost of the generated circuit.
        :param circuit_cost_func: a function that takes as input an instance of
        QuantumGateSequence and returns a float representing the cost of the
        given circuit.
        :param parameters_bounds: bounds for the parameter of the quantum gates
        in the basis. If None, this means that no quantum gate in the basis is
        parametrised. If not all the quantum gates in the basis are
        parametrised, the parameter bounds corresponding to non-parametrised
        quantum gates can take any value.
        """
        self._groups = [
            GateSequenceGroup(basis, objective_unitary, length, p, r,
                              correctness_weight, circuit_cost_weight,
                              circuit_cost_func, parameters_bounds) for _ in
            range(n)]
        self._p = p
        self._length = length
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func
        self._objective_unitary = objective_unitary

    def perform_one_way_crossover(self) -> None:
        """Apply the one way crossover step of the GLOA.

        See the GLOA paper for more precision on this step.
        """
        # One parameter per gate, length gates for each sequences, p sequences.
        number_of_parameters = self._length * self._p
        for group in self._groups:
            # Number of cross-over we will perform on the current group.
            crossover_number = numpy.random.randint(
                number_of_parameters // 2 + 1)
            # Perform the cross-overs.
            for _ in range(crossover_number):
                # Pick at random the indices.
                other_group_index = numpy.random.randint(len(self._groups))
                sequence_index = numpy.random.randint(self._p)
                parameter_index = numpy.random.randint(self._length)
                # Create a new quantum gate sequence from the old one.
                new_sequence = group.sequences[sequence_index]
                new_sequence.params = new_sequence.params.copy()
                # Perform crossover on this sequence.
                new_sequence.params[parameter_index] = \
                    self._groups[other_group_index].sequences[
                        sequence_index].params[parameter_index]
                # Check which sequence is the best.
                old_cost = group.costs[sequence_index]
                new_cost = qdists.gloa_objective_function(new_sequence,
                                                          self._objective_unitary,
                                                          self._correctness_weight,
                                                          self._circuit_cost_weight,
                                                          self._circuit_cost_func)
                if new_cost < old_cost:
                    group.sequences[sequence_index] = new_sequence
            group.update_costs()

    def perform_mutation_and_recombination(self) -> None:
        """Perform the mutation and recombination step on each group."""
        for group in self._groups:
            group.mutate_and_recombine()

    def get_leader(self) -> typing.Tuple[float, QuantumGateSequence]:
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
