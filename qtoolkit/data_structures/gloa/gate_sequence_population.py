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

import typing

import numpy

import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.gloa.gate_sequence_group import GateSequenceGroup
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence


class GateSequencePopulation:

    def __init__(self, basis: typing.Sequence[qtypes.SUdMatrixGenerator],
                 unitary: qtypes.UnitaryMatrix, length: int, n: int, p: int,
                 r: numpy.ndarray, correctness_weight: float,
                 circuit_cost_weight: float, circuit_cost_func: typing.Callable[
            [QuantumGateSequence], float], parameters_bounds: typing.Optional[
            numpy.ndarray] = None) -> None:
        self._groups = [
            GateSequenceGroup(basis, length, p, r, correctness_weight,
                              circuit_cost_weight, circuit_cost_func,
                              parameters_bounds) for _ in range(n)]
        self._p = p
        self._length = length
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func
        self._unitary = unitary

    def perform_one_way_crossover(self) -> None:
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
                old_fidelity = qdists.gloa_fidelity(
                    group.sequences[sequence_index], self._unitary,
                    self._correctness_weight, self._circuit_cost_weight,
                    self._circuit_cost_func)
                new_fidelity = qdists.gloa_fidelity(new_sequence, self._unitary,
                                                    self._correctness_weight,
                                                    self._circuit_cost_weight,
                                                    self._circuit_cost_func)
                if old_fidelity < new_fidelity:
                    group.sequences[sequence_index] = new_sequence

    def perform_mutation_and_recombination(self) -> None:
        for group in self._groups:
            group.mutate_and_recombine(self._unitary)

    def get_leader(self) -> QuantumGateSequence:
        leaders = [group.get_leader(self._unitary) for group in self._groups]

        leader = leaders[0]
        max_fidelity = qdists.gloa_fidelity(leader, self._unitary,
                                            self._correctness_weight,
                                            self._circuit_cost_weight,
                                            self._circuit_cost_func)
        for current_leader in leaders[1:]:
            fidelity = qdists.gloa_fidelity(current_leader, self._unitary,
                                            self._correctness_weight,
                                            self._circuit_cost_weight,
                                            self._circuit_cost_func)
            if fidelity > max_fidelity:
                max_fidelity = fidelity
                leader = current_leader
        return leader
