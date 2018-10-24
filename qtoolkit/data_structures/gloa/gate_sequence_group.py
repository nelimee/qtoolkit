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

"""Implement the GateSequenceGroup class, used to implement GLOA."""

import typing

import numpy

import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence
from qtoolkit.maths.matrix.generation.gate_sequence import \
    generate_random_gate_sequence


class GateSequenceGroup:
    """A group of gate sequences."""

    def __init__(self, basis: typing.Sequence[qtypes.SUdMatrixGenerator],
                 objective_unitary: qtypes.UnitaryMatrix, length: int, p: int,
                 r: numpy.ndarray, correctness_weight: float,
                 circuit_cost_weight: float, circuit_cost_func: typing.Callable[
            [QuantumGateSequence], float], parameters_bounds: typing.Optional[
            numpy.ndarray] = None) -> None:
        """Initialise the GateSequenceGroup instance.

        A GateSequenceGroup is a group of p instances of QuantumGateSequence.

        :param basis: gates available to construct the approximation. Each gate
        can be either a numpy.ndarray (which means that the gate is not
        parametrised) or a callable that takes a float as input and returns a
        numpy.ndarray representing the quantum gate.
        :param objective_unitary: unitary matrix we are trying to approximate.
        :param length: length of the sequences that will be generated.
        :param p: population of the group, i.e. number of gate sequences
        contained in this group.
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
        self._sequences = [
            generate_random_gate_sequence(basis, length, parameters_bounds) for
            _ in range(p)]
        self._basis = basis
        self._r = r
        self._param_bounds = parameters_bounds
        if self._param_bounds is None:
            self._param_bounds = numpy.zeros((2, length))
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func

        self._objective_unitary = objective_unitary
        self._costs = numpy.zeros((p,), dtype=numpy.float)
        self.update_costs()

    def update_costs(self):
        """Update the cached costs.

        This method should be called after one or more sequence(s) of the group
        changed in order to update the cached costs.
        """
        for i in range(len(self._sequences)):
            self._costs[i] = qdists.gloa_objective_function(self._sequences[i],
                                                            self._objective_unitary,
                                                            self._correctness_weight,
                                                            self._circuit_cost_weight,
                                                            self._circuit_cost_func)

    def get_leader(self) -> typing.Tuple[float, QuantumGateSequence]:
        """Get the best sequence of the group.

        The cached costs should be up to date before calling this function. See
        update_costs to know when the cached costs should be updated.
        :return: the best sequence of the group along with its cost.
        """
        idx: int = numpy.argmin(self._costs)
        return self._costs[idx], self._sequences[idx]

    def mutate_and_recombine(self) -> None:
        """Apply the mutate and recombine step of the GLOA.

        See the GLOA paper for more precision on this step.
        """
        # Pre-compute group leader data.
        _, leader = self.get_leader()
        leader_gates = leader.gates
        leader_parameters = leader.params

        # Used to rescale the random values in [0, 1) for the parameters.
        a, b = self._param_bounds[0], self._param_bounds[1]

        # For each member of the group, mutate and recombine it and see if the
        # newly created member is better.
        for seq_idx, sequence in enumerate(self._sequences):
            # Create random gates
            random_gates = numpy.random.randint(0, len(self._basis),
                                                size=len(sequence.gates))
            # Combine the leader's gates, the current sequence's gates and the
            # random gates to mutate the current sequence.
            new_sequence_gates = numpy.array([numpy.random.choice(
                [sequence.gates[i], leader_gates[i], random_gates[i]],
                p=self._r) for i in range(len(sequence.gates))])
            # Create random parameters within the provided bounds.
            random_params = numpy.random.rand(len(sequence.gates)) * (
                b[new_sequence_gates] - a[new_sequence_gates]) + a[
                                new_sequence_gates]
            # Combine all the parameters to create a new set of parameters.
            new_sequence_parameters = sequence.params * self._r[
                0] + leader_parameters * self._r[1] + random_params * self._r[2]

            # Create the new sequence
            new_sequence = QuantumGateSequence(self._basis, new_sequence_gates,
                                               new_sequence_parameters)
            # Save the cost of the current sequence.
            current_sequence_cost = self._costs[seq_idx]
            # Check if the new sequence is better and change it if better.
            new_sequence_cost = qdists.gloa_objective_function(new_sequence,
                                                               self._objective_unitary,
                                                               self._correctness_weight,
                                                               self._circuit_cost_weight,
                                                               self._circuit_cost_func)
            if new_sequence_cost < current_sequence_cost:
                self._sequences[seq_idx] = new_sequence
        self.update_costs()

    @property
    def sequences(self) -> typing.List[QuantumGateSequence]:
        return self._sequences

    @property
    def costs(self):
        return self._costs
