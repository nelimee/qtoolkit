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

from qtoolkit.data_structures.quantum_circuit.quantum_circuit import \
    QuantumCircuit
from qtoolkit.data_structures.quantum_circuit.simplifier.simplification_rule \
    import \
    SimplificationRule


class GateSequenceSimplifier:

    def __init__(self,
                 simplifications: typing.List[SimplificationRule]) -> None:
        self._rules = simplifications

    def add_rule(self, rule: SimplificationRule) -> None:
        self._rules.append(rule)

    def is_simplifiable(self, quantum_circuit: QuantumCircuit) -> bool:
        for qubit_id in range(quantum_circuit.size):
            current_sequence = list(
                quantum_circuit.operations_on_qubit(qubit_id))
            for rule in self._rules:
                if rule.is_simplifiable(current_sequence):
                    return True
        return False

    def is_simplifiable_from_last(self,
                                  quantum_circuit: QuantumCircuit) -> bool:
        last_inserted_op = quantum_circuit.last
        qubits = last_inserted_op.controls
        qubits.append(last_inserted_op.target)
        for qubit_id in qubits:
            current_sequence = list(
                quantum_circuit.operations_on_qubit(qubit_id))
            for rule in self._rules:
                if rule.is_simplifiable_from_last(current_sequence):
                    return True
        return False
