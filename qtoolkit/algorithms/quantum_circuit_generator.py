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

import itertools
import typing

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop
import \
    qtoolkit.data_structures.quantum_circuit.simplifier\
        .gate_sequence_simplifier as qsimpl


def generate_all_quantum_circuits(basis: typing.Sequence[qop.QuantumOperation],
                                  depth: int, qubit_number: int,
                                  simplifier: qsimpl.GateSequenceSimplifier,
                                  include_nodes: bool = False) -> \
    typing.Iterable[qcirc.QuantumCircuit]:
    """Generate all the non-simplifiable quantum circuits.

    This function yields each non-simplifiable (according to the given
    simplifier) sequence of quantum gates picked from the given basis up to the
    given length. If include_nodes is False, all the generated sequence are
    exactly of length depth.

    :param basis: A sequence of basis gates in SU(d) used to construct the
    sequences.
    :param depth: Maximum length of the generated sequences. If include_nodes is
    False, all the generated sequences will have a length equal to depth. Else,
    the sequences will have a length lower or equal to depth.
    :param simplifier: The simplifier used to know if a given quantum circuit is
    simplifiable or not.
    :param include_nodes: If False, only sequences of the specified depth are
    generated. Else, all sequences of length between 1 and depth (inclusive) are
    generated.
    :return: An iterable yielding QuantumCircuit instance.
    """
    # 1. Check inputs
    assert depth > 0, "Depth should be 1 or more."
    assert len(basis) > 1, "The basis should contain at least 2 matrices."
    assert qubit_number > 0, "Negative number of qubit is impossible."

    quantum_circuit = qcirc.QuantumCircuit(qubit_number)

    return _generate_all_quantum_circuits_impl(basis, depth, quantum_circuit,
                                               simplifier, include_nodes)


def _generate_all_quantum_circuits_impl(
    basis: typing.Sequence[qop.QuantumOperation], depth: int,
    quantum_circuit: qcirc.QuantumCircuit,
    simplifier: qsimpl.GateSequenceSimplifier, include_nodes: bool = False):
    """Generate all the non-simplifiable quantum circuits.

    :param basis: A sequence of basis gates in SU(d) used to construct the
    sequences.
    :param depth: Maximum length of the generated sequences. If include_nodes is
    False, all the generated sequences will have a length equal to depth. Else,
    the sequences will have a length lower or equal to depth.
    :param simplifier: The simplifier used to know if a given quantum circuit is
    simplifiable or not.
    :param include_nodes: If False, only sequences of the specified depth are
    generated. Else, all sequences of length between 1 and depth (inclusive) are
    generated.
    :return: An iterable yielding QuantumCircuit instance.
    """
    if depth == 1:
        for op in basis:
            for variant in _generate_all_variations(op, quantum_circuit.size):
                quantum_circuit.add_operation(variant)
                if not simplifier.is_simplifiable_from_last(quantum_circuit):
                    yield quantum_circuit
                quantum_circuit.pop()
        return

    for op in basis:
        for variant in _generate_all_variations(op, quantum_circuit.size):
            quantum_circuit.add_operation(variant)
            if not simplifier.is_simplifiable_from_last(quantum_circuit):
                for circ in _generate_all_quantum_circuits_impl(basis,
                                                                depth - 1,
                                                                quantum_circuit,
                                                                simplifier,
                                                                include_nodes):
                    yield circ
            quantum_circuit.pop()


def _generate_all_variations(op: qop.QuantumOperation, qubit_number: int):
    """Generate all the possible variations for one quantum gate.

    In short, apply the given gate in all the possible manners and yield each
    possible manner as a QuantumOperation instance.

    :param op: the considered operation.
    :param qubit_number: the number of qubits of the overall circuit.
    """
    assert len(op.controls) < 2, "Multi-controlled gates are not supported."

    no_control = (len(op.controls) == 0)

    for trgt in range(qubit_number):
        if no_control:
            yield qop.QuantumOperation(op.gate, trgt)
        else:
            for ctrl in itertools.chain(range(trgt),
                                        range(trgt + 1, qubit_number)):
                yield qop.QuantumOperation(op.gate, trgt, [ctrl])
