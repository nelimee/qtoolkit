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

"""Implementation of a generator of quantum circuits.

This module contains a unique function :py:func:`generate_all_quantum_circuits`
that generate *all* the quantum circuits satisfying some conditions.

For random generation of quantum circuits see
:py:func:`~qtoolkit.maths.matrix.generation.quantum_circuit.generate_random_quantum_circuit`.
"""

import itertools
import typing

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.data_structures.quantum_circuit.quantum_operation as qop
import qtoolkit.data_structures.quantum_circuit.simplifier.quantum_circuit_simplifier as qsimpl


def generate_all_quantum_circuits(
    basis: typing.Sequence[qop.QuantumOperation],
    depth: int,
    qubit_number: int,
    simplifier: qsimpl.QuantumCircuitSimplificationDetector,
    include_nodes: bool = False,
    return_progress: bool = False,
) -> typing.Iterable[
    typing.Union[qcirc.QuantumCircuit, typing.Tuple[qcirc.QuantumCircuit, int]]
]:
    """Generate all the non-simplifiable quantum circuits.

    This function yields each non-simplifiable (according to the given
    `simplifier`) sequence of quantum gates picked from the given basis up to
    the given `depth`. If `include_nodes` is False, all the generated sequence
    are exactly of length `depth`.

    :param basis: A sequence of
        :py:class:`~.QuantumOperation` used as atomic blocks. All the operations
        in the generated quantum circuits are ensured to be in `basis`.
        :py:class:`~.QuantumOperation` instances in the sequence can be either
        "abstract" or not (see :py:class:`~.QuantumOperation` documentation for
        a definition of an "abstract" quantum operation).
    :param depth: Maximum length of the generated sequences. If `include_nodes`
        is False, all the generated sequences will have a length equal to
        `depth`. Else, the sequences will have a length lower or equal to
        `depth`.
    :param qubit_number: the number of qubits on which the generated circuits
        will act. This number should be sufficiently large to be able to apply
        the operations in `basis`.
    :param simplifier: The simplifier used to know if a given quantum circuit is
        simplifiable or not.
    :param include_nodes: If False, only sequences of the specified `depth` are
        generated. Else, all sequences of length between 1 and `depth`
        (inclusive) are generated.
    :param return_progress: If False, the function returns only the generated
        quantum circuits. If True, the function returns tuples containing the
        generated quantum circuit along with its ID.
    :return: An iterable yielding :py:class:`~.QuantumCircuit` or
        (:py:class:`~.QuantumCircuit`, int) instances.
    """
    # 1. Check inputs
    assert depth > 0, "Depth should be 1 or more."
    assert len(basis) > 1, "The basis should contain at least 2 matrices."
    assert qubit_number > 0, "Negative number of qubit is impossible."

    quantum_circuit = qcirc.QuantumCircuit(qubit_number)

    # From https://stackoverflow.com/questions/952914/making-a-flat-list-out
    # -of-list-of-lists-in-python#45323085
    variations = list(
        itertools.chain.from_iterable(
            (_generate_all_variations(op, qubit_number) for op in basis)
        )
    )

    if return_progress:
        return _gen_all_qcircs_progress_impl(
            depth, quantum_circuit, simplifier, variations, include_nodes
        )
    else:
        return _gen_all_qcircs_impl(
            depth, quantum_circuit, simplifier, variations, include_nodes
        )


def _gen_all_qcircs_progress_impl(
    depth: int,
    qc: qcirc.QuantumCircuit,
    simplifier: qsimpl.QuantumCircuitSimplificationDetector,
    variations: typing.List[qop.QuantumOperation],
    include_nodes: bool = False,
    progress: int = 0,
):
    """Generate all the non-simplifiable quantum circuits.

    :param depth: Maximum length of the generated sequences. If `include_nodes`
        is False, all the generated sequences will have a length equal to
        `depth`. Else, the sequences will have a length lower or equal to
        `depth`.
    :param qc: The quantum circuit representing the current state of the
        generation.
    :param simplifier: The simplifier used to know if a given quantum circuit is
        simplifiable or not.
    :param variations: All the possible operations that can be applied on one
        level of the quantum circuit (one level <=> depth=constant).
    :param include_nodes: If False, only sequences of the specified `depth` are
        generated. Else, all sequences of length between 1 and `depth`
        (inclusive) are generated.
    :return: An iterable yielding (:py:class:`~.QuantumCircuit`, int).
    """

    if include_nodes:
        yield qc, progress

    if depth == 1:
        for variant in variations:
            qc.add_operation(variant)
            progress += 1
            if not simplifier.is_simplifiable_from_last(qc):
                yield qc, progress
            qc.pop()
        return

    for variant in variations:
        qc.add_operation(variant)
        progress += 1
        if not simplifier.is_simplifiable_from_last(qc):
            yield from _gen_all_qcircs_progress_impl(
                depth - 1, qc, simplifier, variations, include_nodes, progress
            )
        else:
            # We pruned some circuit, we need to update the progress
            # accordingly.
            progress += len(variations) ** (depth - 1)
        qc.pop()


def _gen_all_qcircs_impl(
    depth: int,
    qc: qcirc.QuantumCircuit,
    simplifier: qsimpl.QuantumCircuitSimplificationDetector,
    variations: typing.List[qop.QuantumOperation],
    include_nodes: bool = False,
):
    """Generate all the non-simplifiable quantum circuits.

    :param depth: Maximum length of the generated sequences. If `include_nodes`
        is False, all the generated sequences will have a length equal to
        `depth`. Else, the sequences will have a length lower or equal to
        `depth`.
    :param qc: The quantum circuit representing the current state of the
        generation.
    :param simplifier: The simplifier used to know if a given quantum circuit is
        simplifiable or not.
    :param variations: All the possible operations that can be applied on one
        level of the quantum circuit (one level <=> depth=constant).
    :param include_nodes: If False, only sequences of the specified `depth` are
        generated. Else, all sequences of length between 1 and `depth`
        (inclusive) are generated.
    :return: An iterable yielding :py:class:`~.QuantumCircuit`.
    """

    if include_nodes:
        yield qc

    if depth == 1:
        for variant in variations:
            qc.add_operation(variant)
            if not simplifier.is_simplifiable_from_last(qc):
                yield qc
            qc.pop()
        return

    for variant in variations:
        qc.add_operation(variant)
        if not simplifier.is_simplifiable_from_last(qc):
            yield from _gen_all_qcircs_impl(
                depth - 1, qc, simplifier, variations, include_nodes
            )
        qc.pop()


def _generate_all_variations(op: qop.QuantumOperation, qubit_number: int):
    """Generate all the possible variations for one quantum gate.

    In short, apply the given gate in all the possible manners and yield each
    possible manner as a :py:class:`~.QuantumOperation` instance.

    :param op: The considered operation. Can be "abstract" or not (see
        :py:class:`~.QuantumOperation` documentation for a definition of an
        "abstract" quantum operation).
    :param qubit_number: the number of qubits of the overall circuit.
    """
    assert len(op.controls) < 2, "Multi-controlled gates are not supported."

    no_control = len(op.controls) == 0

    for trgt in range(qubit_number):
        if no_control:
            yield qop.QuantumOperation(op.gate, trgt)
        else:
            for ctrl in itertools.chain(range(trgt), range(trgt + 1, qubit_number)):
                yield qop.QuantumOperation(op.gate, trgt, [ctrl])
