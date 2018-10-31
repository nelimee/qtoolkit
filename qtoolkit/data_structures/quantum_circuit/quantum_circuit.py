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

import networkx as nx
import numpy

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgate
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop


class QuantumCircuit:

    def __init__(self, qubit_number: int) -> None:

        assert qubit_number > 0, "A circuit with less than 1 qubit cannot be " \
                                 "created."

        self._qubit_number = qubit_number
        self._graph = nx.DiGraph()
        self._node_counter = 0

        for i in range(qubit_number):
            self._graph.add_node(self._node_counter, type="input")
            self._node_counter += 1

        self._last_inserted_operations = list(range(qubit_number))

    def add_operation(self, operation: qop.QuantumOperation) -> None:

        self._check_operation(operation)
        current_node_ID = self._node_counter
        self._graph.add_node(self._node_counter, type="op", op=operation)
        self._node_counter += 1

        # Create the target wire
        self._graph.add_edge(self._last_inserted_operations[operation.target],
                             current_node_ID)
        self._last_inserted_operations[operation.target] = current_node_ID

        # Create the control wires
        for ctrl in operation.controls:
            self._graph.add_edge(self._last_inserted_operations[ctrl],
                                 current_node_ID)
            self._last_inserted_operations[ctrl] = current_node_ID

    def apply(self, gate: qgate.QuantumGate, target: int,
              controls: typing.Sequence[int] = ()) -> None:

        self.add_operation(qop.QuantumOperation(gate, target, controls))

    def _check_operation(self, operation: qop.QuantumOperation) -> None:
        if operation.target >= self._qubit_number or operation.target < 0:
            raise IndexError(
                f"The operation's target ({operation.target}) is not valid "
                f"for the current quantum circuit with {self._qubit_number} "
                f"qubits.")
        for ctrl in operation.controls:
            if ctrl >= self._qubit_number or ctrl < 0:
                raise IndexError(
                    "One of the control qubit is not valid for the current "
                    "quantum circuit.")

    def pop(self) -> qop.QuantumOperation:
        if self._node_counter <= self._qubit_number:
            raise RuntimeError(
                "Attempting to pop a QuantumOperation from an empty "
                "QuantumCircuit.")
        op = self._graph.nodes[self._node_counter - 1]['op']
        self._graph.remove_node(self._node_counter - 1)
        self._node_counter -= 1
        return op

    @property
    def operations(self):
        return (self._graph.nodes[i]['op'] for i in
                range(self._qubit_number, self._node_counter))

    @property
    def matrix(self) -> numpy.ndarray:
        ret = numpy.identity(2 ** self._qubit_number)
        for operation in self.operations:
            ret = ret @ operation.matrix(self._qubit_number)
        return ret

    @property
    def size(self):
        return self._qubit_number
