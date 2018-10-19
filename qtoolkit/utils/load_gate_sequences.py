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

import os.path
import typing

import numpy

import qtoolkit.algorithms.quantum_gate_sequences_generator as sud_gen
import qtoolkit.utils.types as qtypes


def load_gate_sequences(basis: typing.Sequence[qtypes.SUdMatrix], depth: int,
                        simplifiable_sequences: typing.Iterable[
                            typing.Sequence[int]],
                        basis_str: typing.Sequence[str] = None,
                        include_nodes: bool = False) -> typing.Tuple[
    numpy.ndarray, numpy.ndarray]:
    """Getter for the SU(d) pre-computed sequences.

    This function will try to load the sequences corresponding to the given
    parameters from the qtoolkit/data/ directory. If the sequences are not
    found within this directory, the function will compute the sequences, save
    them in the directory for further reuse and return them to the caller.

    :param basis: The basis used to compute gates.
    :param basis_str: A list of string identifiers that will identify each gate
    in the basis provided. If None, the function will not try to load the
    sequences from a pre-computed file and will regenerate them.
    :param depth: The number of gates we want in our sequences.
    :param simplifiable_sequences: A collection of simplifiable sequences. Each
    simplifiable sequence is a sequence of indices each representing a matrix in
    the given basis. If [0, 1, 0] is in simplifiable_sequence, this means that
    basis[0] @ basis[1] @ basis[0] can be simplified to a shorter sequence of
    gates from the basis.
    :param include_nodes: If False, only sequences of the specified depth are
    generated. Else, all sequences of length between 1 and depth (inclusive) are
    generated.
    :return: Two numpy array of dimensions (N, 3) and (N, depth) with N
    the number of sequences generated.
    The first array stores the SO(3) vectors representing the quantum gates
    obtained with sequences of gates in the basis of length depth.
    The second array stores the indices of the quantum gates in the basis
    used to construct the corresponding SO(3) vector.
    With so3, sequences = load_so3_filling(...), the following is always
    valid:
    For each 0 <= i < N:
        matrix = basis[0]
        for j in range(1, depth):
            matrix = matrix @ basis[sequences[i][j]]
        assert numpy.allclose(su2_trans.so3_to_su2(so3[i]), matrix)
    """
    # Some constants
    this_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(this_dir, os.path.pardir)
    data_dir = os.path.realpath(os.path.join(parent_dir, "data"))

    sud_matrices, gate_sequences = None, None

    if basis_str is not None:
        npz_file = os.path.join(data_dir, f"{'_'.join(basis_str)}_{depth}"
                                          f"{'_wn' if include_nodes else 'nn'}"
                                          f".npz")
        if os.path.isfile(npz_file):
            # If the SO(3) vectors were already computed, load them.
            with numpy.load(npz_file) as data:
                matrices = data["matrices"]
                gate_sequences = data["gate_sequences"]

    # Construct the nearest neighbour structure
    if sud_matrices is None or gate_sequences is None:
        gate_id_type = numpy.uint8
        if len(basis) > 2 ** 8 - 1:
            gate_id_type = numpy.uint16
        elif len(basis) > 2 ** 16 - 1:
            gate_id_type = numpy.uint32
        gate_sequences = list()
        matrices = list()
        for gate_sequence, matrix in sud_gen.generate_all_gate_sequences(basis,
                                                                         depth,
                                                                         simplifiable_sequences,
                                                                         gate_id_type=gate_id_type,
                                                                         include_nodes=include_nodes):
            matrices.append(matrix.copy())
            gate_sequences.append(gate_sequence.copy())

        matrices = numpy.array(matrices, dtype=float)
        gate_sequences = numpy.array(gate_sequences, dtype=int)
        if basis_str is not None:
            numpy.savez(npz_file, matrices=matrices,
                        gate_sequences=gate_sequences)

    return matrices, gate_sequences
