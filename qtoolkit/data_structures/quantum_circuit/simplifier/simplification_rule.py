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

"""Implementation of several simplification rules.

A simplification rule implements a test to check if the given
:py:class:`~.QuantumCircuit` is simplifiable according to the rule it
implements.
This module implements the base class for all the simplification rules:
:py:class:`~.SimplificationRule`. All the derived classes should implement at
least one of the two methods to check for simplifiability:

1. :py:meth:`~.SimplificationRule.is_simplifiable`
2. :py:meth:`~.SimplificationRule.is_simplifiable_from_last`

.. note::
    The method :py:meth:`.SimplificationRule.is_simplifiable` is used only when
    :py:meth:`.QuantumCircuitSimplificationDetector.is_simplifiable` is used.

    Similarly, the method
    :py:meth:`.SimplificationRule.is_simplifiable_from_last` is used only when
    :py:meth:`.QuantumCircuitSimplificationDetector.is_simplifiable_from_last`
    is used.
"""

import numpy

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgate
import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc


class SimplificationRule:
    """Representation of a generic simplification rule."""

    def is_simplifiable(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        """Check if the given quantum circuit is simplifiable.

        :param quantum_circuit: the :py:class:`~.QuantumCircuit` to check for
            simplifiability.
        :return: True if `quantum_circuit` is simplifiable according to the
            rules stored, else False.
        """
        raise NotImplementedError("This method should be overridden.")

    def is_simplifiable_from_last(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        """Check if the last part of `quantum_circuit` is simplifiable.

        This method can be used to check if the last gate of the quantum circuit
        introduced a simplification or not.

        :param quantum_circuit: the quantum circuit to check for
            simplifiability.
        :return: True if `quantum_circuit` is simplifiable according to the
            rules stored, else False.
        """
        raise NotImplementedError("This method should be overridden.")


class InverseRule(SimplificationRule):
    def __init__(self, quantum_gate: qgate.QuantumGate) -> None:
        self._gate = quantum_gate
        self._inverse = quantum_gate.H

    def is_simplifiable_from_last(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        if quantum_circuit.size < 2:
            return False

        last = quantum_circuit.last
        opgen = quantum_circuit.get_n_last_operations_on_qubit(2, last.target)
        op_names = [op.gate.name for op in opgen]
        if len(op_names) < 2:
            return False

        result = self._gate.name == op_names[0] and self._inverse.name == op_names[1]
        result = result or (
            self._inverse.name == op_names[0] and self._gate.name == op_names[1]
        )
        return result


class CXInverseRule(SimplificationRule):
    def is_simplifiable_from_last(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        if quantum_circuit.size < 2:
            return False

        last = quantum_circuit.last
        operations = list(
            quantum_circuit.get_n_last_operations_on_qubit(2, last.target)
        )
        if len(operations) < 2:
            return False

        op1, op2 = operations[0], operations[1]

        result = True
        result = result and (op1.gate.name == "X" and op2.gate.name == "X")
        result = result and (op1.target == op2.target)
        result = result and (
            len(op1.controls) == 1
            and len(op2.controls) == 1
            and op1.controls[0] == op2.controls[0]
        )
        return result


class RepeatedRule(SimplificationRule):
    def __init__(self, quantum_gate: qgate.QuantumGate, repetition: int) -> None:
        self._gate = quantum_gate
        self._repetition = repetition

    def is_simplifiable_from_last(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        if quantum_circuit.size < self._repetition:
            return False

        last = quantum_circuit.last
        opgen = quantum_circuit.get_n_last_operations_on_qubit(
            self._repetition, last.target
        )
        op_names = [op.gate.name for op in opgen]
        if len(op_names) < self._repetition:
            return False

        return all((name == self._gate.name for name in op_names))


class NearIdentityRule(SimplificationRule):
    def __init__(self, atol: float = 1e-8, rtol: float = 1e-5):
        self._atol = atol
        self._rtol = rtol

    def is_simplifiable_from_last(self, quantum_circuit: qcirc.QuantumCircuit) -> bool:
        matrix = quantum_circuit.matrix
        dim = matrix.shape[0]
        return numpy.allclose(
            matrix, numpy.identity(dim), rtol=self._rtol, atol=self._atol
        )
