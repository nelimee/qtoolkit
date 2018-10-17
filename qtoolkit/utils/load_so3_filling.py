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

import qtoolkit.data_structures.quantum_gate_lazy_generator as seq_gen
import qtoolkit.maths.matrix.su2.transformations as su2_trans
import qtoolkit.utils.types as qtypes


def load_so3_filling(basis: typing.Sequence[qtypes.SU2Matrix], basis_str: str,
                     depth: int, simplifiable_sequences: typing.Set[bytes]) -> \
    typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """Getter for the SO(3) (or SU(2)) pre-computed sequences.

    :param basis: The basis used to compute gates.
    :param basis_str: A string representing the basis. This string is used
    to search for already-computed sequences in files and to save the
    computed sequences to a file.
    :param depth: The number of gates we want in our sequences.
    :param simplifiable_sequences: A set of simplifications that can be
    applied to sequences of gates in the provided basis.
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
    npz_file = os.path.join(data_dir, f"{basis_str}_{depth}.npz")

    # Construct the nearest neighbour structure
    if not os.path.isfile(npz_file):
        # If we did not computed the SO(3) vectors for the moment, compute
        # and save them.
        lazy_trie = seq_gen.QuantumGateLazyGenerator(basis, depth,
                                                     simplifiable_sequences)

        gate_sequences_indices = list()
        so3_vectors = list()
        for gate_sequence, su2_matrix in \
            lazy_trie.generate_all_possible_unitaries():
            so3_vectors.append(su2_trans.su2_to_so3(su2_matrix))
            gate_sequences_indices.append(gate_sequence.copy())

        so3_vectors = numpy.array(so3_vectors, dtype=float)
        gate_sequences_indices = numpy.array(gate_sequences_indices, dtype=int)
        numpy.savez(npz_file, so3_vectors=so3_vectors,
                    gate_sequences_indices=gate_sequences_indices)

    else:
        # If the SO(3) vectors were already computed, load them.
        with numpy.load(npz_file) as data:
            so3_vectors = data["so3_vectors"]
            gate_sequences_indices = data["gate_sequences_indices"]

    return so3_vectors, gate_sequences_indices
