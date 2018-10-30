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

from typing import Optional, Sequence, List

import numpy

import qtoolkit.utils.types as qtypes


class QuantumGateSequence:

    def __init__(self, basis: Sequence[qtypes.SUdMatrixGenerator],
                 sequence: numpy.ndarray,
                 parameters: Optional[numpy.ndarray] = None,
                 resulting_matrix: qtypes.SUdMatrix = None,
                 dimension: int = None) -> None:
        """

        :param basis:
        :param sequence:
        :param parameters:
        :param resulting_matrix:
        :param dimension:
        """
        self._basis = basis
        self._sequence = sequence
        self._parameters = parameters
        if self._parameters is None:
            self._parameters = numpy.zeros((len(sequence),), dtype=numpy.float)
        self._resulting_matrix = resulting_matrix
        if dimension is not None:
            self._dim = dimension
        elif resulting_matrix is not None:
            self._dim = resulting_matrix.shape[0]
        else:
            self._dim = self._get_sud_matrix(0).shape[0]

    def _get_sud_matrix(self, gate_position: int) -> qtypes.SUdMatrix:
        if callable(self._basis[self._sequence[gate_position]]):
            return self._basis[self._sequence[gate_position]](
                self._parameters[gate_position])
        else:
            return self._basis[self._sequence[gate_position]]

    @property
    def matrix(self) -> qtypes.UnitaryMatrix:
        if self._resulting_matrix is None:
            self._resulting_matrix = self._get_sud_matrix(0)
            for gate_position in range(1, len(self._sequence)):
                self._resulting_matrix = self._resulting_matrix @ \
                                         self._get_sud_matrix(
                    gate_position)
        return self._resulting_matrix

    @property
    def gates(self) -> numpy.ndarray:
        return self._sequence

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def params(self) -> Optional[List[Optional[qtypes.QuantumGateParameter]]]:
        return self._parameters

    @params.setter
    def params(self, value) -> None:
        self._parameters = value

    def __matmul__(self, other: 'QuantumGateSequence') -> 'QuantumGateSequence':
        if self._resulting_matrix is None and other._resulting_matrix is None:
            matrix = None
        else:
            # If at least one of the two matrices has already been computed then
            # it is more efficient to compute the other matrix.
            matrix = self.matrix @ other.matrix

        sequence = numpy.concatenate((self._sequence, other._sequence))
        parameters = numpy.concatenate((self.params, other.params))
        return QuantumGateSequence(self._basis, sequence, parameters, matrix,
                                   self._dim)


class InvertibleQuantumGateSequence(QuantumGateSequence):

    def __init__(self, basis: Sequence[qtypes.SUdMatrixGenerator],
                 sequence: numpy.ndarray, inverses: numpy.ndarray,
                 parameters: Optional[numpy.ndarray] = None,
                 inverse_parameters: Optional[Sequence[
                     Optional[qtypes.GateParameterTransformation]]] = None,
                 resulting_matrix: qtypes.SUdMatrix = None,
                 dimension: int = None) -> None:
        super().__init__(basis, sequence, parameters, resulting_matrix,
                         dimension)
        self._inverses = inverses

        def do_nothing(x: qtypes.QuantumGateParameter):
            return x

        if inverse_parameters is None:
            self._inverse_parameters = [do_nothing] * len(basis)
        else:
            self._inverse_parameters = [inv or do_nothing for inv in
                                        inverse_parameters]

    @property
    def H(self) -> 'InvertibleQuantumGateSequence':
        inverted_parameters = [self._inverse_parameters[i](param) for i, param
                               in enumerate(reversed(self._parameters))]
        inverted_sequence = self._inverses[self._sequence][::-1]
        inv_matrix = None
        if self._resulting_matrix is not None:
            inv_matrix = self._resulting_matrix.T.conj()
        return InvertibleQuantumGateSequence(self._basis, inverted_sequence,
                                             self._inverses,
                                             numpy.array(inverted_parameters),
                                             self._inverse_parameters,
                                             inv_matrix,
                                             self._dim)

    def __matmul__(self,
                   other: 'InvertibleQuantumGateSequence') -> \
        'InvertibleQuantumGateSequence':
        # Using super().__matmul__().
        res = super().__matmul__(other)
        return InvertibleQuantumGateSequence(self._basis, res.gates,
                                             self._inverses, res.params,
                                             self._inverse_parameters,
                                             res._resulting_matrix, self.dim)
