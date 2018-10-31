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

import qtoolkit.utils.constants.matrices as mconsts
from qtoolkit.data_structures.quantum_circuit.gate_hierarchy import QuantumGate


class QuantumOperation:

    def __init__(self, gate: QuantumGate, target: int,
                 controls: typing.Sequence[int] = tuple()) -> None:
        assert target not in controls, "The target qubit cannot be used as a " \
                                       "control qubit."
        self._gate = gate
        self._target = target
        self._controls = controls

    @property
    def gate(self):
        return self._gate

    @property
    def controls(self):
        return self._controls

    @property
    def target(self):
        return self._target

    def matrix(self, qubit_number: int) -> numpy.ndarray:
        # If the operation is not a controlled by any qubit then we can
        # simplify greatly the algorithm.
        if not self._controls:
            ret = 1
            for qubit_index in range(qubit_number):
                # If we are on the target qubit then apply the gate.
                if qubit_index == self._target:
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


def control(operation: QuantumOperation, *controls: int) -> QuantumOperation:
    assert operation.target not in controls, "The target qubit cannot be used" \
                                             " as a control qubit."
    return QuantumOperation(operation.gate, operation.target,
                            tuple(set(operation.controls) | set(controls)))
