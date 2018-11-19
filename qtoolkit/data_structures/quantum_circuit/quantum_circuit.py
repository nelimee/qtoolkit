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

"""Implementation of the QuantumCircuit class."""

import copy
import typing

import networkx as nx
import numpy

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgate
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop


class QuantumCircuit:

    def __init__(self, qubit_number: int, cache_matrix: bool = True) -> None:
        """Initialise QuantumCircuit instances.

        :param qubit_number: the number of qubits in the circuit.
        """
        assert qubit_number > 0, "A circuit with less than 1 qubit cannot be " \
                                 "created."

        self._qubit_number = qubit_number
        self._graph = nx.MultiDiGraph()
        self._node_counter = 0

        for qubit_id in range(qubit_number):
            self._graph.add_node(self._node_counter, type="input", key=qubit_id)
            self._node_counter += 1

        self._last_inserted_operations = numpy.arange(qubit_number)
        self._cache_matrix = cache_matrix
        self._matrix = None
        if self._cache_matrix:
            self._matrix = numpy.identity(2 ** self._qubit_number)

    def add_operation(self, operation: qop.QuantumOperation) -> None:
        """Add an operation to the circuit.

        :param operation: the operation to add to the QuantumCircuit instance.
        """
        self._check_operation(operation)
        current_node_id = self._node_counter
        self._graph.add_node(self._node_counter, type="op", op=operation)
        self._node_counter += 1

        # Create the target wire
        self._create_edge(self._last_inserted_operations[operation.target],
                          current_node_id, operation.target)
        self._last_inserted_operations[operation.target] = current_node_id

        # Create the control wires
        for ctrl in operation.controls:
            self._create_edge(self._last_inserted_operations[ctrl],
                              current_node_id, ctrl)
            self._last_inserted_operations[ctrl] = current_node_id

        # Compute the new matrix if needed and possible.
        if self._cache_matrix:
            self._matrix = self._matrix @ operation.matrix(self._qubit_number)

    def apply(self, gate: qgate.QuantumGate, target: int,
              controls: typing.Sequence[int] = ()) -> None:
        """Apply a quantum operation to the circuit.

        :param gate: the quantum gate to apply.
        :param target: the qubit to apply the operation on.
        :param controls: the control qubit(s).
        """
        self.add_operation(qop.QuantumOperation(gate, target, controls))

    def _check_operation(self, operation: qop.QuantumOperation) -> None:
        """Check if the operation is valid. If not, raise an exception.

        :param operation: the operation to check for validity.
        :raise IndexError: if the qubits of the operation (target or control(s))
        are not within the range of the current instance.
        """
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
        """Delete the last inserted operation and return it.

        :return: the last inserted operation.
        """
        if self._node_counter <= self._qubit_number:
            raise RuntimeError(
                "Attempting to pop a QuantumOperation from an empty "
                "QuantumCircuit.")
        # Recover the last operation performed.
        op = self.last
        # Update the last_inserted structure
        for pred, _, key in self._graph.in_edges(nbunch=self._node_counter - 1,
                                                 keys=True):
            self._last_inserted_operations[key] = pred
        # Remove the node (and the edges associated to it).
        self._graph.remove_node(self._node_counter - 1)
        self._node_counter -= 1
        # Compute the new matrix if needed and possible.
        if self._cache_matrix:
            self._matrix = self._matrix @ op.matrix(self._qubit_number).T.conj()
        return op

    def _create_edge(self, from_id: int, to_id: int, qubit_id: int) -> None:
        self._graph.add_edge(from_id, to_id, key=qubit_id)

    def get_n_last_operations_on_qubit(self, n: int, qubit_id: int):
        return list(self.get_n_last_operations_on_qubit_reversed(n, qubit_id))[
               ::-1]

    def get_n_last_operations_on_qubit_reversed(self, n: int, qubit_id: int):
        """Returns an iterable on the n last operations performed on the qubit.

        If there is not enough operations, returns all the operations.

        :param n: number of operations to look for.
        :param qubit_id: the qubit of interest.
        :return: an iterable of at most n items.
        """
        current = self._last_inserted_operations[qubit_id]
        while n > 0 and current >= self.qubit_number:
            yield self._graph.nodes[current]['op']
            n -= 1
            # Update the current node.
            current = next(filter(
                lambda node_id: qubit_id in self._graph.get_edge_data(node_id,
                                                                      current),
                self._graph.predecessors(current)))

    def get_operations_on_qubit_reversed(self, qubit_id: int):
        # Ask for node_counter operations. The get_n_last_operations_... will
        # return node_counter operations if possible or all the operations on
        # the given qubit.
        return self.get_n_last_operations_on_qubit_reversed(self._node_counter,
                                                            qubit_id)

    def get_operations_on_qubit(self, qubit_id: int):
        # Ask for node_counter operations. The get_n_last_operations_... will
        # return node_counter operations if possible or all the operations on
        # the given qubit.
        return self.get_n_last_operations_on_qubit(self._node_counter, qubit_id)

    def __getitem__(self, idx: int) -> qop.QuantumOperation:
        return self._graph.nodes[idx + self._qubit_number]['op']

    @property
    def last(self):
        if self._node_counter > self._qubit_number:
            return self._graph.nodes[self._node_counter - 1]['op']
        else:
            return None

    @property
    def operations(self):
        """Getter on the operations performed in this quantum circuit.

        :return: a generator that generates all the operations of the circuit.
        """
        return (self._graph.nodes[i]['op'] for i in
                range(self._qubit_number, self._node_counter))

    def operations_on_qubit(self, qubit_index: int):
        """Getter for the operations applied on the qubit at the given index.

        Warning: for the moment this method does not use fully the graph
        structure of the QuantumCircuit class and so iterate on all the quantum
        operations.

        :param qubit_index: the qubit we are interested in.
        """
        return filter(lambda op: op.target == qubit_index, self.operations)

    def gates_on_qubit(self, qubit_index: int):
        """Getter for the gates applied on the qubit at the given index.

        Warning: for the moment this method does not use fully the graph
        structure of the QuantumCircuit class and so iterate on all the quantum
        operations.

        :param qubit_index: the qubit we are interested in.
        :return:
        """
        return (op.gate for op in self.operations_on_qubit(qubit_index))

    @property
    def matrix(self) -> numpy.ndarray:
        """Getter on the unitary matrix representing the circuit.

        The matrix is re-computed each time the property is called.

        :return: the unitary matrix representing the current quantum circuit.
        """
        if self._cache_matrix:
            return self._matrix
        ret = numpy.identity(2 ** self._qubit_number)
        for operation in self.operations:
            ret = ret @ operation.matrix(self._qubit_number)
        return ret

    @property
    def qubit_number(self):
        """Getter on the number of qubits of the current instance."""
        return self._qubit_number

    @property
    def size(self):
        return self._node_counter - self._qubit_number

    def __iadd__(self, other: 'QuantumCircuit') -> 'QuantumCircuit':
        """Add all the operations contained in other to the current instance.

        :param other: the quantum circuit containing the operations to append
        to the current instance.
        :return: The union of self and other.
        """
        # 1. Checks
        if self.qubit_number != other.qubit_number:
            raise RuntimeError(f"The number of qubits of the first circuit "
                               f"({self.qubit_number}) does not match the "
                               f"number of qubits of the second circuit "
                               f"({other.qubit_number}).")
        # 2. Update the graph
        # 2.1. First remove the "input" nodes from the other graph. We don't
        # want to change or copy the other graph so we take a view of the other
        # graph without the "input" nodes.
        other_subgraph = other._graph.subgraph(
            range(other.qubit_number, other._node_counter))
        # 2.2. Regroup the two graphs into one graph.
        self._graph = nx.disjoint_union(self._graph, other_subgraph)
        # 2.3. Join the nodes if possible.
        for qubit_index in range(self.qubit_number):
            old_neighbor = list(other._graph.neighbors(qubit_index))
            if old_neighbor:
                new_neighbor = old_neighbor[
                                   0] - other.qubit_number + self._node_counter
                self._graph.add_edge(
                    self._last_inserted_operations[qubit_index], new_neighbor)
                # Only change the last inserted index if we joined the nodes.
                self._last_inserted_operations[qubit_index] = new_neighbor
        # 3. Update the other attributes:
        self._node_counter += other._node_counter - other.qubit_number
        if self._cache_matrix and other._matrix is not None:
            self._matrix = self.matrix @ other.matrix

        return self

    def __matmul__(self: 'QuantumCircuit',
                   other: 'QuantumCircuit') -> 'QuantumCircuit':
        cpy = copy.copy(self)
        return cpy.__iadd__(other)

    def __copy__(self) -> 'QuantumCircuit':
        cpy = QuantumCircuit(self._qubit_number,
                             cache_matrix=self._cache_matrix)
        if self.compressed:
            cpy._compressed_graph = copy.copy(self._compressed_graph)
        else:
            cpy._graph = self._graph.copy()
        cpy._node_counter = self._node_counter
        cpy._last_inserted_operations = self._last_inserted_operations.copy()
        if self._cache_matrix:
            cpy._matrix = self._matrix
        return cpy

    def compress(self) -> 'QuantumCircuit':
        if not self.compressed:
            self._compressed_graph = CompressedMultiDiGraph(self._graph)
            del self._graph
        return self

    def uncompress(self) -> 'QuantumCircuit':
        if self.compressed:
            self._graph = self._compressed_graph.uncompress()
            del self._compressed_graph
        return self

    @property
    def compressed(self) -> bool:
        return hasattr(self, '_compressed_graph')

    def inverse(self) -> 'QuantumCircuit':
        inv = QuantumCircuit(self._qubit_number,
                             cache_matrix=self._cache_matrix)
        for op in reversed(list(self.operations)):
            inv.add_operation(op.inverse())

        return inv

    def __str__(self) -> str:
        return '\n'.join(("{Cs}{opname} {controls}, {target}".format(
            Cs="C" * len(op.controls), opname=op.gate.name,
            controls=','.join(map(str, op.controls)), target=op.target) for op
        in self.operations))


class CompressedMultiDiGraph:

    def __init__(self, graph: nx.MultiDiGraph = None) -> None:
        if graph is None:
            self._qubit_number = 0
            return

        node_number = len(graph.nodes)
        edge_number = len(graph.edges)

        if node_number < 2 ** 8:
            data_type = numpy.uint8
        elif node_number < 2 ** 16:
            data_type = numpy.uint16
        else:
            data_type = numpy.uint32

        # We keep each edge with its corresponding qubit ID.
        self._from_arr = numpy.zeros((edge_number,), dtype=data_type)
        self._to_arr = numpy.zeros((edge_number,), dtype=data_type)
        self._data_arr = numpy.zeros((edge_number,), dtype=data_type)
        for idx, (u, v, qubit_id) in enumerate(graph.edges):
            self._from_arr[idx] = u
            self._to_arr[idx] = v
            self._data_arr[idx] = qubit_id

        # And the we keep each node
        self._qubit_number = 0
        self._is_op_node = numpy.zeros((node_number,), dtype=numpy.bool)
        self._operations = list()
        for node_id, node_data in graph.nodes.items():
            if node_data['type'] == 'op':
                self._is_op_node[node_id] = True
                self._operations.append(node_data['op'])
            else:
                self._qubit_number += 1

    def __copy__(self) -> 'CompressedMultiDiGraph':
        cpy = CompressedMultiDiGraph()
        cpy._qubit_number = self._qubit_number
        cpy._from_arr = self._from_arr.copy()
        cpy._to_arr = self._to_arr.copy()
        cpy._data_arr = self._data_arr.copy()
        cpy._is_op_node = self._is_op_node.copy()
        cpy._operations = copy.copy(self._operations)
        return cpy

    def uncompress(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        if self._qubit_number == 0:
            return graph

        # Re-create the nodes.
        for i in range(self._qubit_number):
            graph.add_node(i, type="input", key=i)

        for node_id in range(self._qubit_number, len(self._is_op_node)):
            graph.add_node(node_id, type="op",
                           op=self._operations[node_id - self._qubit_number])

        # Re-create the edges
        for u, v, qubit_id in zip(self._from_arr, self._to_arr, self._data_arr):
            graph.add_edge(u, v, key=qubit_id)

        return graph


CircuitCostFunction = typing.Callable[[QuantumCircuit], float]
