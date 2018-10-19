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

import qtoolkit.utils.types as qtypes


def generate_all_gate_sequences(basis: typing.Sequence[qtypes.SUdMatrix],
                                depth: int,
                                simplifiable_sequences: typing.Iterable[
                                    typing.Sequence[int]],
                                gate_id_type: qtypes.UnsignedIntegerType =
                                numpy.uint8,
                                include_nodes: bool = False) -> typing.Iterable[
    typing.Tuple[numpy.ndarray, qtypes.SUdMatrix]]:
    """Generate all the non-simplifiable sequences of gates.

    This function yields each non-simplifiable (according to the given
    simplifiable_sequences structure) sequence of matrices picked from the given
    basis up to the given length. If include_nodes is False, all the generated
    sequence are exactly of length depth.

    :param basis: A sequence of basis gates in SU(d) used to construct the
    sequences.
    :param depth: Maximum length of the generated sequences. If include_nodes is
    False, all the generated sequences will have a length equal to depth. Else,
    the sequences will have a length lower or equal to depth.
    :param simplifiable_sequences: A collection of simplifiable sequences. Each
    simplifiable sequence is a sequence of indices each representing a matrix in
    the given basis. If [0, 1, 0] is in simplifiable_sequence, this means that
    basis[0] @ basis[1] @ basis[0] can be simplified to a shorter sequence of
    gates from the basis.
    :param gate_id_type: Type used to store indices representing a gate in the
    basis. If the basis has less than 256 gates (which is likely to happen), the
    default value will work fine. The provided type should be able to store
    integers up to len(basis).
    :param include_nodes: If False, only sequences of the specified depth are
    generated. Else, all sequences of length between 1 and depth (inclusive) are
    generated.
    :return: An iterable yielding 2 values:
        - a 1D numpy.ndarray containing integers representing a sequence of
          gates picked from the given basis.
        - the matrix obtained by multiplying each matrix in the sequence
          described in 1.
    """
    gate_id_type = numpy.dtype(gate_id_type)
    # 1. Check inputs
    assert depth > 0, "Depth should be 1 or more."
    assert len(basis) > 1, "The basis should contain at least 2 matrices."
    assert len(basis) < (2 ** (8 * gate_id_type.itemsize)), (
        f"The basis should contain at most {2**(8*gate_id_type.itemsize)} "
        f"matrices. Try to change the parameter gate_id_type to a larger "
        f"type to prevent this exception to occur.")
    identity_dxd = numpy.identity(basis[0].shape[0])
    for gate in basis:
        assert numpy.allclose(identity_dxd, gate @ gate.T.conj()), (
            "All the gates in the basis should be in SU(d).")
    for simplifiable_sequence in simplifiable_sequences:
        for gate_idx in simplifiable_sequence:
            assert gate_idx < len(basis), (
                "The gates indices in the simplifiable sequences should all "
                "represent a gate in the basis provided.")

    # 2. Create structures that will be used in the following parts.
    # Simplification rules in a structure adapted for quick searching.
    simplifications: set = {numpy.array(seq, dtype=gate_id_type).tobytes() for
                            seq in simplifiable_sequences}
    max_simpl_length = max(map(len, simplifiable_sequences))
    # Variables representing the current position in the complete tree of gate
    # sequences.
    # Invariants: at all time during the tree traversal:
    #    1) current_gate_sequence[:current_depth+1] is valid and represents the
    #  current position in the tree.
    #    2) current_gate_sequence[current_depth+1:] is filled with zeros.
    #    3) saved_matrices[:current_depth+1] is valid and stores the matrix
    #  represented by the gate sequence current_gate_sequence[:current_depth+1].
    #    4) saved_matrices[current_depth+1:] is filled with zeros.
    dim = basis[0].shape[0]
    current_depth = -1
    max_depth = depth - 1
    current_gate_sequence = numpy.zeros((depth,), dtype=gate_id_type)
    saved_matrices = numpy.zeros((depth, dim, dim), dtype=numpy.complex)

    # 3. Do the tree traversal and yield a result at each leaf. Also yields
    #  at each node if include_nodes == True.

    # Python does not have a do-while statement, so we need to duplicate code
    # here.
    # First, go to the first node to initialise everything.
    current_depth = _go_to_next_non_simplifiable_node(current_depth,
                                                      current_gate_sequence,
                                                      saved_matrices, basis,
                                                      simplifications,
                                                      max_simpl_length)

    # Then go through the tree until we go back to the root.
    while current_depth != -1:
        # Check if we need to yield the current node.
        if current_depth == max_depth or include_nodes:
            yield (current_gate_sequence[:current_depth + 1],
                   saved_matrices[current_depth])
        # update the current depth.
        current_depth = _go_to_next_non_simplifiable_node(current_depth,
                                                          current_gate_sequence,
                                                          saved_matrices, basis,
                                                          simplifications,
                                                          max_simpl_length)


def _go_to_next_non_simplifiable_node(current_depth: int,
                                      current_gate_sequence: numpy.ndarray,
                                      saved_matrices: numpy.ndarray,
                                      basis: typing.Sequence[qtypes.SUdMatrix],
                                      simplifications: typing.Set[bytes],
                                      max_simplification_length: int) -> int:
    """Go to the next non simplifiable node.

    :param current_depth: The current depth in the tree.
    :param current_gate_sequence: The current state of the traversal.
    :param saved_matrices: The array of saved matrices. The matrices are updated
    in place.
    :param basis: A sequence of basis gates. Used to update the saved_matrices.
    :param simplifications: A set containing all the sequences that can be
    simplified and should not be included in the final result.
    :param max_simplification_length: The maximum length up to which the
    function will check for possible simplifications.
    :return: The new current_depth. The other parameters are mutable and so
    are updated in place, we don't need to return them.
    """
    # First, go to the next node.
    current_depth = _go_to_next_node(current_depth, current_gate_sequence,
                                     saved_matrices, basis)
    # Then, while the current node is simplifiable, cut branches and update
    # the current node.
    while _is_simplifiable(current_depth, current_gate_sequence,
                           simplifications, max_simplification_length):
        # The current node is simplifiable. This means that we don't want to
        # explore any of its children because they will all be simplifiable.
        current_depth = _cut_current_node_and_go_next(current_depth,
                                                      current_gate_sequence,
                                                      saved_matrices, basis)

    # We are at the next simplifiable node, just return.
    return current_depth


def _go_to_next_node(current_depth: int, current_gate_sequence: numpy.ndarray,
                     saved_matrices: numpy.ndarray,
                     basis: typing.Sequence[qtypes.SUdMatrix]) -> int:
    """Go to the next node and update the matrices.

    :param current_depth: The current depth in the tree.
    :param current_gate_sequence: The current state of the traversal.
    :param saved_matrices: The array of saved matrices. The matrices are updated
    in place.
    :param basis: A sequence of basis gates. Used to update the saved_matrices.
    :return: The new current_depth. The other parameters are mutable and so
    are updated in place, we don't need to return them.
    """
    max_depth = current_gate_sequence.shape[0] - 1

    # If we can go down and explore deeper levels, then go!
    if current_depth < max_depth:
        current_depth += 1
        _update_saved_matrices(current_depth, current_gate_sequence,
                               saved_matrices, basis)
        return current_depth

    # Else, we are on a leaf. We go up until going right is possible.
    return _go_up_and_move_right(current_depth, current_gate_sequence,
                                 saved_matrices, basis)


def _cut_current_node_and_go_next(current_depth: int,
                                  current_gate_sequence: numpy.ndarray,
                                  saved_matrices: numpy.ndarray,
                                  basis: typing.Sequence[
                                      qtypes.SUdMatrix]) -> int:
    """Go to the next node that is not a children of the current node.

    :param current_depth: The current depth in the tree.
    :param current_gate_sequence: The current state of the traversal.
    :param saved_matrices: The array of saved matrices. The matrices are updated
    in place.
    :param basis: A sequence of basis gates. Used to update the saved_matrices.
    :return: The new current_depth. The other parameters are mutable and so
    are updated in place, we don't need to return them.
    """
    # First we try to go right.
    if current_gate_sequence[current_depth] < len(basis) - 1:
        current_gate_sequence[current_depth] += 1
        _update_saved_matrices(current_depth, current_gate_sequence,
                               saved_matrices, basis)
        return current_depth

    # If going right is not possible, we go up until we can go right and perform
    # the move to the right.
    return _go_up_and_move_right(current_depth, current_gate_sequence,
                                 saved_matrices, basis)


def _go_up_and_move_right(current_depth: int,
                          current_gate_sequence: numpy.ndarray,
                          saved_matrices: numpy.ndarray,
                          basis: typing.Sequence[qtypes.SUdMatrix]) -> int:
    # We go up until we can go right or find the root.
    while current_gate_sequence[current_depth] == len(
        basis) - 1 and current_depth != -1:
        saved_matrices[current_depth, :, :] = 0
        current_gate_sequence[current_depth] = 0
        current_depth -= 1

    # Move right if we are not at the root.
    if current_depth != -1:
        current_gate_sequence[current_depth] += 1
        _update_saved_matrices(current_depth, current_gate_sequence,
                               saved_matrices, basis)

    # Return the current depth.
    return current_depth


def _update_saved_matrices(current_depth: int,
                           current_gate_sequence: numpy.ndarray,
                           saved_matrices: numpy.ndarray,
                           basis: typing.Sequence[qtypes.SUdMatrix]) -> None:
    """Update the saved matrices when a move has been performed.

    :param current_depth: The current depth in the tree.
    :param current_gate_sequence: The current state of the traversal.
    :param saved_matrices: The array of saved matrices. The matrices are updated
    in place.
    :param basis: A sequence of basis gates. Used to update the saved_matrices.
    :return: Nothing. The parameters are mutable and so are updated in place.
    """
    # Special case when the current depth is 0: we change the first matrix.
    if current_depth == 0:
        saved_matrices[current_depth] = basis[
            current_gate_sequence[current_depth]]
    else:
        saved_matrices[current_depth] = saved_matrices[current_depth - 1] @ \
                                        basis[current_gate_sequence[
                                            current_depth]]


def _is_simplifiable(current_depth: int, current_gate_sequence: numpy.ndarray,
                     simplifications: typing.Set[bytes],
                     max_simplification_length: int) -> bool:
    """Check is the given gate sequence is simplifiable.

    This function assumes that the given gate_sequence has already been checked
    for simplifiability up to the penultimate gate. This means that the only
    possible source of simplifiability is the last gate of the provided
    sequence.

    :param current_depth: The depth of the current state.
    :param current_gate_sequence: The current state in the tree traversal.
    :param simplifications: A set containing all the simplifiable sequences.
    :param max_simplification_length: Maximum length of simplifiable sequences.
    Simplifications will not be searched above this length.
    :return: True if the sequence is simplifiable, else False.
    """
    max_len = min(max_simplification_length, current_depth)
    for current_simplification_length in range(1, max_len + 1):
        start = current_depth - current_simplification_length
        end = current_depth
        if current_gate_sequence[start:end + 1].tobytes() in simplifications:
            return True
    return False
