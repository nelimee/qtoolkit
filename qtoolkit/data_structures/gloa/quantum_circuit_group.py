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

"""Implement the :py:class:`~.QuantumCircuitGroup` class used in GLOA.

The :py:class:`~.QuantumCircuitGroup` class represents what the `original paper\
 <https://arxiv.org/abs/1004.2242>`_ call a "group". It is a collection of
entities (in this particular case "entities" refers to "instances of
:py:class:`~.QuantumCircuit`") admitting a leader.
"""

import copy
import typing

import numpy

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop
import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.maths.matrix.generation.quantum_circuit as qc_gen
import qtoolkit.utils.types as qtypes


class QuantumCircuitGroup:
    """A group of :py:class:`~.QuantumCircuit`.

    The instances of :py:class:`~.QuantumCircuit` are grouped into an instance
    of :py:class:`~.QuantumCircuitGroup` to factorise the code.
    """

    def __init__(self, basis: typing.Sequence[qop.QuantumOperation],
                 objective_unitary: qtypes.UnitaryMatrix, length: int, p: int,
                 r: numpy.ndarray, correctness_weight: float,
                 circuit_cost_weight: float,
                 circuit_cost_func: qcirc.CircuitCostFunction,
                 parameters_bounds: qtypes.Bounds = None) -> None:
        """Initialise the :py:class:`~.QuantumCircuitGroup` instance.

        A :py:class:`~.QuantumCircuitGroup` is a group composed of `p` instances
        of :py:class:`~.QuantumCircuit`.

        :param basis: a sequence of allowed operations. The operations can be
            "abstract" (i.e. with None entries, see the documentation for the
            :py:class:`~.QuantumOperation` class) or not (i.e. with specified
            entries).
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
            :py:class:`~.QuantumCircuit` and returns a float representing the
            cost of the given circuit.
        :param parameters_bounds: a list of bounds for each operation in the
            `basis`. A None value in this list means that the corresponding
            operation is not parametrised. A None value for the whole list
            (default value) means that no gate in `basis` is parametrised.
        """
        self._qubit_number = objective_unitary.shape[0].bit_length() - 1
        self._circuits = [
            qc_gen.generate_random_quantum_circuit(self._qubit_number, basis,
                                                   length, parameters_bounds)
            for _ in range(p)]
        self._basis = basis
        self._r = r
        self._length = length
        self._param_bounds = parameters_bounds
        if self._param_bounds is None:
            self._param_bounds = [None] * self._qubit_number
        self._correctness_weight = correctness_weight
        self._circuit_cost_weight = circuit_cost_weight
        self._circuit_cost_func = circuit_cost_func

        self._objective_unitary = objective_unitary
        self._costs = numpy.zeros((p,), dtype=numpy.float)
        self._update_costs()

    def _update_costs(self):
        """Update the cached costs.

        This method should be called after one or more sequence(s) of the group
        changed in order to update the cached costs.
        """
        for i in range(len(self._circuits)):
            self._costs[i] = qdists.gloa_objective_function(self._circuits[i],
                                                            self._objective_unitary,
                                                            self._correctness_weight,
                                                            self._circuit_cost_weight,
                                                            self._circuit_cost_func)

    def get_leader(self) -> typing.Tuple[float, qcirc.QuantumCircuit]:
        """Get the best quantum circuit of the group.

        :return: the best sequence of the group along with its cost.
        """
        idx: int = numpy.argmin(self._costs)
        return self._costs[idx], self._circuits[idx]

    def mutate_and_recombine(self) -> None:
        """Apply the mutate and recombine step of the GLOA.

        See the `GLOA paper <https://arxiv.org/abs/1004.2242>`_ for more
        precision on this step.
        """
        # Pre-compute group leader data.
        _, leader = self.get_leader()

        # For each member of the group, mutate and recombine it and see if the
        # newly created member is better.
        for seq_idx, current in enumerate(self._circuits):
            new_circuit = qcirc.QuantumCircuit(self._qubit_number,
                                               cache_matrix=True)
            random = qc_gen.generate_random_quantum_circuit(self._qubit_number,
                                                            self._basis,
                                                            self._length,
                                                            self._param_bounds)

            for ops in zip(current.operations, leader.operations,
                           random.operations):
                new_circuit.add_operation(self._combine_operations(ops))

            new_cost = qdists.gloa_objective_function(new_circuit,
                                                      self._objective_unitary,
                                                      self._correctness_weight,
                                                      self._circuit_cost_weight,
                                                      self._circuit_cost_func)
            if new_cost < self._costs[seq_idx]:
                self._circuits[seq_idx] = new_circuit
                self._costs[seq_idx] = new_cost
        return

    def _combine_operations(self, operations: typing.Sequence[
        qop.QuantumOperation]) -> qop.QuantumOperation:
        """Combine the 3 given operations into one operation.

        The combined operation is randomly chosen from the 3 given operations
        with the probability distribution `r` given at the instance construction
        and then randomly mutated with characteristics of the other operations.

        :param operations: A sequence of 3 :py:class:`~.QuantumOperation`.
        :return: a random merge of the 3 given operations.
        """
        op1, op2, op3 = operations[0], operations[1], operations[2]

        new_operation = copy.copy(numpy.random.choice(operations, p=self._r))
        control_number = len(new_operation.controls)
        new_operation.controls = []
        new_operation.target = numpy.random.choice(
            [op1.target, op2.target, op3.target], p=self._r)

        while len(new_operation.controls) < control_number:
            ctrl = numpy.random.randint(0, self._qubit_number)
            if ctrl != new_operation.target and ctrl not in \
                new_operation.controls:
                new_operation.controls.append(ctrl)

        if new_operation.is_parametrised():
            raise NotImplementedError(
                "Parametrised operations are not supported for the moment.")
        return new_operation

    @property
    def circuits(self) -> typing.List[qcirc.QuantumCircuit]:
        """Getter for the stored list of :py:class:`~.QuantumCircuit`."""
        return self._circuits

    @property
    def costs(self):
        """Getter for the pre-computed costs."""
        return self._costs
