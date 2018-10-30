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

from collections import deque
from typing import List, Union, Tuple

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgates
import qtoolkit.utils.constants.quantum_gates as qgconsts


class QuantumCircuit:

    def __init__(self, qubit_number: int) -> None:
        assert qubit_number > 0, "The circuit should have at least 1 qubit."
        self._last_inserted_index: Union[int, Tuple[int, int]] = None
        self._qubits = [list() for _ in range(qubit_number)]
        self._insertion_order = deque()

    def apply(self, quantum_gate: qgates.QuantumGate,
              qubits: List[int]) -> None:
        assert len(qubits) == 1 or (
            len(qubits) == 2 and quantum_gate.name == 'CX'), (
            "The QuantumCircuit class only supports one-qubit gates and CX.")
        assert quantum_gate.dim == 2 ** len(qubits), (
            "The quantum gate dimension does not match the number of qubits "
            "given.")
        for qubit in qubits:
            if not 0 <= qubit < len(self._qubits):
                raise IndexError(
                    f"The qubit n°{qubit} is out of range for the current "
                    f"quantum circuit.")
        self._insertion_order.append(qubits)
        if len(qubits) == 1:
            # If it is a 1-qubit gate.
            idx = qubits[0]
            self._last_inserted_index = idx
            self._qubits[idx].append(quantum_gate)
        else:
            # Else it is CX
            ctrl, trgt = qubits[0], qubits[1]
            self._last_inserted_index = (ctrl, trgt)
            self._qubits[ctrl].append(qgconsts.CX_ctrl)
            self._qubits[trgt].append(qgconsts.CX_trgt)

    def remove_last_inserted(self) -> None:
        if not self._insertion_order:
            raise RuntimeError(
                "Impossible to remove the last inserted gate as there is no "
                "gate in the circuit.")
        for qubit_idx in self._insertion_order.pop():
            self._qubits[qubit_idx].pop()

    def get_last_modified_qubits(self) -> Union[
        Tuple[List[qgates.QuantumGate]], Tuple[
            List[qgates.QuantumGate], List[qgates.QuantumGate]]]:
        if self._last_inserted_index is None:
            raise RuntimeError(
                "[QuantumCircuit.get_last_modified_qubits] Attempting to "
                "acces the last modified qubit of a quantum circuit but no "
                "operation as been performed on this circuit.")
        if isinstance(self._last_inserted_index, int):
            return self._qubits[self._last_inserted_index],
        else:
            return (self._qubits[self._last_inserted_index[0]],
                    self._qubits[self._last_inserted_index[1]])

    @property
    def size(self):
        return len(self._qubits)

    @property
    def qubits(self):
        return self._qubits
