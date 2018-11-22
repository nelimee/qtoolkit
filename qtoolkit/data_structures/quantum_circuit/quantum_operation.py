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

"""Implementation of the QuantumOperation class."""

import copy
import typing
from typing import Optional

import numpy

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as gh
import qtoolkit.utils.constants.matrices as mconsts


class QuantumOperation:

    def __init__(self,
                 gate: typing.Union[gh.QuantumGate, gh.ParametrisedQuantumGate],
                 target: Optional[int] = None,
                 controls: Optional[typing.Sequence[Optional[int]]] = None,
                 parameters: Optional[
                     typing.Sequence[Optional[float]]] = None) -> None:
        """A class representing a quantum operation.

        For the moment, the QuantumOperation class only support 1-qubit gates
        with arbitrary controls. In the real world, this is not really a huge
        limitation as most of the quantum hardware only supports 1-qubit gates
        and one controlled operation like CX.

        If any of the value of controls or the value of target are None, this
        means that the operation is "abstract", i.e. it represent the general
        operation, not applied to a particular case.
        For example, QuantumOperation(X, None, [None, None]) represents the
        doubly-controlled X operation.

        :param gate: the 1-qubit quantum gate of the operation.
        :param target: the target qubit of the given quantum gate.
        :param controls: an arbitrary number of control qubits. If `controls`
            is None, this means that the gate is not controlled by any qubit.
        :param parameters: parameters of the given gate. A value of None can
            mean 2 things: if the provided gate is a ParametrisedQuantumGate,
            then the operation represents an "abstract" operation, else it
            is just a reminder that the stored gate is not parametrised.
        """
        if controls is None:
            controls = list()
        assert target is None or target not in controls, (
            "The target qubit cannot be used as a control qubit.")
        self._gate = gate
        self._target = target
        self._controls = controls
        self._parameters = parameters

    def __copy__(self) -> 'QuantumOperation':
        return QuantumOperation(self._gate, self._target,
                                copy.copy(self._controls),
                                copy.copy(self._parameters))

    @property
    def gate(self):
        """Getter for the 1-qubit quantum gate stored in this operation."""
        if self.is_parametrised():
            return self._gate(self._parameters)
        return self._gate

    @property
    def target(self) -> int:
        """Getter for the target qubit."""
        return self._target

    @property
    def controls(self) -> Optional[typing.Sequence[Optional[int]]]:
        """Getter for the control qubits."""
        return self._controls

    @target.setter
    def target(self, value: int) -> None:
        assert value not in self._controls
        self._target = value

    @controls.setter
    def controls(self, value: typing.Iterable[int]) -> None:
        assert self.target not in value
        self._controls = list(value)

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        if self._parameters is not None:
            return numpy.array(self._parameters)
        return self._parameters

    @parameters.setter
    def parameters(self, value: numpy.ndarray) -> None:
        self._parameters = value

    def matrix(self, qubit_number: int) -> numpy.ndarray:
        """Computes the matrix representation of the operation.

        :param qubit_number: the number of qubits of the overall circuit.
        :return: a 2**qubit_number x 2**qubit_number matrix representing the
        current operation.
        """
        # If the operation is not a controlled by any qubit then we can
        # simplify greatly the algorithm.
        if not self._controls:
            ret = 1
            for qubit_index in range(qubit_number):
                # If we are on the target qubit then apply the gate.
                if qubit_index == self._target:
                    if self.is_parametrised():
                        ret = numpy.kron(ret,
                                         self._gate(self.parameters).matrix)
                    else:
                        ret = numpy.kron(ret, self._gate.matrix)
                # Else, we should multiply by the gate that is controlled.
                else:
                    ret = numpy.kron(ret, mconsts.ID2)
            return ret

        # Else, we have control qubits.
        ret = numpy.zeros((2 ** qubit_number, 2 ** qubit_number),
                          dtype=numpy.complex)
        # For each possible values for the control qubits.
        for ctrl_values in range(2 ** len(self._controls)):
            ctrl_values_list = [(ctrl_values >> k) & 1 for k in
                                range(len(self._controls))]
            current_control_index = 0
            current_matrix = 1
            for qubit_index in range(qubit_number):
                # If we are on the target qubit then...
                if qubit_index == self._target:
                    # If all the control qubits are 1, then multiply by the
                    # gate.
                    if all(ctrl_values_list):
                        if self.is_parametrised():
                            m = self._gate(self.parameters).matrix
                            current_matrix = numpy.kron(current_matrix, m)
                        else:
                            current_matrix = numpy.kron(current_matrix,
                                                        self._gate.matrix)
                    # Else, we should multiply by the gate that is controlled.
                    else:
                        current_matrix = numpy.kron(current_matrix, mconsts.ID2)
                # Else if we are on a control qubit, determine if we should
                # use P0 or P1 and apply it.
                elif qubit_index in self._controls:
                    current_matrix = numpy.kron(current_matrix,
                                                mconsts.P1 if ctrl_values_list[
                                                    current_control_index]
                                                else mconsts.P0)
                    current_control_index += 1
                # Else, the current qubit do nothing.
                else:
                    current_matrix = numpy.kron(current_matrix, mconsts.ID2)
            ret += current_matrix
        return ret

    def inverse_inplace(self) -> 'QuantumOperation':
        # TODO: H property may not be defined if self._gate is a
        # ParametrisedQuantumGate
        self._gate = self._gate.H
        return self

    def inverse(self) -> 'QuantumOperation':
        cpy = copy.copy(self)
        return cpy.inverse_inplace()

    @property
    def H(self):
        return self.inverse()

    def is_parametrised(self):
        return callable(self._gate)

    def __call__(self, *args, **kwargs) -> 'QuantumOperation':
        return QuantumOperation(self._gate(*args, **kwargs), self.target,
                                self.controls, args)


def control(operation: QuantumOperation, *controls: int) -> QuantumOperation:
    assert operation.target not in controls, "The target qubit cannot be used" \
                                             " as a control qubit."
    return QuantumOperation(operation.gate, operation.target,
                            tuple(set(operation.controls) | set(controls)))
