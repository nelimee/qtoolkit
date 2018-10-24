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
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence
from qtoolkit.maths.matrix.generation.gate_sequence import \
    generate_random_gate_sequence


class GateSequenceGroup:

    def __init__(self, basis: typing.Sequence[qtypes.SUdMatrixGenerator],
                 length: int, p: int, r: numpy.ndarray,
                 correctness_weight: float, circuit_cost_weight: float,
                 circuit_cost_func: typing.Callable[
                     [QuantumGateSequence], float],
                 parameters_bounds: typing.Optional[
                     numpy.ndarray] = None) -> None:
        """Initialise the GateSequenceGroup instance.

        :param basis: basis used for the computations.
        basis. If the gate is parametrised, inverse_parameters is also needed.
        :param length: length of the sequences that will be generated.
        :param p: population of the group, i.e. number of gate sequences
        contained in this group.
        :param r: mutation parameters. Should be an array of 3 floats.
        :param parameters_bounds: bounds used for the parameters. Irrelevant
        values (i.e. associated to a gate which is not parametrised) can be set
        to any value: they will not be used.
        """
        self._sequences = [
            generate_random_gate_sequence(basis, length, parameters_bounds) for
            _ in range(p)]
        self._basis = basis
        self._current_leader: QuantumGateSequence = None
        self._r = r
        self._param_bounds = parameters_bounds
        if self._param_bounds is None:
            self._param_bounds = numpy.zeros((2, length))
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func

    def _compute_leader(self, objective_unitary: qtypes.UnitaryMatrix) -> None:
        best_sequence = self._sequences[0]
        maximum_fidelity = qdists.gloa_fidelity(best_sequence,
                                                objective_unitary,
                                                self._correctness_weight,
                                                self._circuit_cost_weight,
                                                self._circuit_cost_func)

        for sequence in self._sequences:
            fidelity = qdists.gloa_fidelity(sequence, objective_unitary,
                                            self._correctness_weight,
                                            self._circuit_cost_weight,
                                            self._circuit_cost_func)
            if fidelity > maximum_fidelity:
                maximum_fidelity = fidelity
                best_sequence = sequence
        self._current_leader = best_sequence

    def get_leader(self,
                   objective_unitary: qtypes.UnitaryMatrix) -> \
        QuantumGateSequence:
        self._compute_leader(objective_unitary)
        return self._current_leader

    def mutate_and_recombine(self,
                             objective_unitary: qtypes.UnitaryMatrix) -> None:
        # Pre-compute group leader data.
        leader = self._current_leader or self.get_leader(objective_unitary)
        leader_gates = leader.gates
        leader_parameters = leader.params

        # Used to rescale the random values in [0, 1) for the parameters.
        a, b = self._param_bounds[0], self._param_bounds[1]

        # For each member of the group, mutate and recombine it and see if the
        # newly created member is better.
        for seq_idx, sequence in enumerate(self._sequences):
            # Save the fidelity of the current sequence.
            # TODO: this is a duplicate computation because this value has
            # TODO: been computed in self.get_leader(). Maybe add memoization.
            current_sequence_fidelity = qdists.gloa_fidelity(sequence,
                                                             objective_unitary,
                                                             self._correctness_weight,
                                                             self._circuit_cost_weight,
                                                             self._circuit_cost_func)
            # Create random gates
            random_gates = numpy.random.randint(0, len(self._basis),
                                                size=len(sequence.gates))
            # Combine the leader's gates, the current sequence's gates and the
            # random gates to mutate the current sequence.
            new_sequence_gates = numpy.array([numpy.random.choice(
                [sequence.gates[i], leader_gates[i], random_gates[i]],
                p=self._r) for i in range(len(sequence.gates))])
            # Create random parameters within the provided bounds.
            random_params = numpy.random.rand(len(sequence.gates)) * (b - a) + a
            # Combine all the parameters to create a new set of parameters.
            new_sequence_parameters = sequence.params * self._r[
                0] + leader_parameters * self._r[1] + random_params * self._r[2]

            # Create the new sequence
            new_sequence = QuantumGateSequence(self._basis, new_sequence_gates,
                                               new_sequence_parameters)
            # Check if the new sequence is better and change it if better.
            new_sequence_fidelity = qdists.gloa_fidelity(new_sequence,
                                                         objective_unitary,
                                                         self._correctness_weight,
                                                         self._circuit_cost_weight,
                                                         self._circuit_cost_func)
            if new_sequence_fidelity > current_sequence_fidelity:
                self._sequences[seq_idx] = new_sequence

    @property
    def sequences(self) -> typing.List[QuantumGateSequence]:
        return self._sequences
