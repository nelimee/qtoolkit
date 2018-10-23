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

"""This module declare the QuantumGateSequence class.

The QuantumGateSequence class is a wrapper around a sequence of quantum gates.
It behaves like a numpy matrix, but also stores the sequence of quantum gates
that produced the matrix.
"""

import typing

import numpy

import qtoolkit.utils.types as qtypes


class QuantumGateSequence:

    def __init__(self, basis: typing.Sequence[qtypes.SUdMatrix],
                 gate_sequence: numpy.ndarray, inverses: numpy.ndarray = None,
                 resulting_matrix: qtypes.SUdMatrix = None) -> None:
        """Initialise QuantumGateSequence instances.

        :param basis: List of quantum gates that are combined to produce the
        resulting matrix.
        :param gate_sequence: List of indices that represent the sequence of
        quantum gates from the basis that compose the resulting matrix.
        :param inverses: Array of integers mapping each gate in the given basis
        to its inverse in the same basis. inverses[i] = j means that
        basis[i] @ basis[j] = basis[j] @ basis[i] = I.
        :param resulting_matrix: The resulting matrix. If not None, the given
        matrix is assumed to be correct and is not checked. If None, the matrix
        is re-computed lazily from the basis and the gate_sequence inputs.
        """
        # The basis needs to contain at least 2 elements.
        assert len(basis) >= 2

        # Storing constants in the instance.
        self._d = basis[0].shape[0]

        # Storing the provided parameters.
        self._basis = basis
        self._gate_sequence = gate_sequence.copy()
        if resulting_matrix is None:
            self._matrix = None
        else:
            self._matrix = resulting_matrix.copy()

        if inverses is None:
            self._inverses = None
        else:
            self._inverses = inverses

    @property
    def matrix(self):
        """Computes if necessary and returns the matrix represented.

        The matrix represented is computed only once and then stored for
        further re-use.

        :return: the represented matrix.
        """
        if self._matrix is None:
            self._matrix = self._basis[self._gate_sequence[0]]
            for idx in self._gate_sequence[1:]:
                self._matrix = self._matrix @ self._basis[idx]
        return self._matrix

    @property
    def gates(self):
        """Getter for the underlying gate sequence."""
        return self._gate_sequence

    @property
    def inverses(self):
        """Getter for the inverses."""
        if self._inverses is None:
            self._compute_inverses()
        return self._inverses

    @property
    def dimension(self):
        """Getter for the dimension of the current QuantumGateSequence."""
        return self._d

    def inverse(self) -> 'QuantumGateSequence':
        """Inverse the quantum gate sequence.

        Inverting the quantum gate sequence is done by inverting each gate
        in the sequence separately and then apply those inverses in reverse
        order.

        :return: A new quantum gate sequence representing the inverse.
        """
        self._compute_inverses()

        # Inverting a matrix is O(n^3), which is the same as the complexity
        # of multiplying a matrix by another matrix. So we perform the
        # inversion now if the matrix is available
        if self._matrix is not None:
            resulting_matrix = numpy.linalg.inv(self._matrix)
        else:
            resulting_matrix = None
        # We return a new QuantumGateSequence representing the self.inverse().
        return QuantumGateSequence(self._basis,
                                   self._inverses[self._gate_sequence][::-1],
                                   self._inverses,
                                   resulting_matrix=resulting_matrix)

    def _compute_inverses(self) -> None:
        """Find the inverse of each gate in the provided basis and store them.

        The inverses should be themselves in the basis! Moreover, if the
        inverses have already been computed, this function does nothing.
        Inverses are stored in a vector whose structure is presented in
        the parameter documentation of the __init__ method.
        """
        IDENTITY = numpy.identity(self.dimension, dtype=complex)
        if self._inverses is not None and self._basis_contains_inverses():
            return
        # Compute which gate is the inverse of which one in the basis.
        self._inverses = numpy.zeros((len(self._basis),), dtype=numpy.int) - 1
        for idx, gate in enumerate(self._basis):
            # If we don't know the inverse of this gate
            if self._inverses[idx] == -1:
                for jdx, possible_inverse in enumerate(self._basis):
                    if numpy.allclose(IDENTITY, gate @ possible_inverse):
                        self._inverses[idx] = jdx
                        self._inverses[jdx] = idx
        self._check_basis_contains_inverses()

    def _basis_contains_inverses(self) -> bool:
        """Check if each gate in the basis has its inverse in the basis.

        This method should not be called before _compute_inverses.
        """
        return numpy.all(self._inverses != -1)

    def _check_basis_contains_inverses(self):
        """Check if each gate in the basis has its inverse in the basis.

        :raise RuntimeError: if there is a gate that does not have its
        inverse in the basis.

        This method should not be called before _compute_inverses.
        """
        if not self._basis_contains_inverses():
            raise RuntimeError("The basis provided is not complete. Each gate"
                               " should have its inverse in the basis.")

    def __matmul__(self, other: 'QuantumGateSequence') -> 'QuantumGateSequence':
        """Wrapper around the numpy.ndarray.__matmul__ function.

        :param other: An other QuantumGateSequence.
        :return: The QuantumGateSequence resulting of the matrix multiplication
        between self and other.
        """
        # Checking that the basis are the same.
        # This is non-negligible in terms of computation time for the
        # Solovay-Kitaev algorithm, so I remove it for the moment.
        # assert len(self._basis) == len(other._basis)
        # for self_basis_gate, other_basis_gate in zip(self._basis,
        # other._basis):
        #     assert numpy.allclose(self_basis_gate, other_basis_gate)

        # Returning the result:
        return QuantumGateSequence(self._basis, numpy.concatenate(
            (self._gate_sequence, other._gate_sequence)), self._inverses,
                                   self.matrix @ other.matrix)
