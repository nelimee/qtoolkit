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

"""Test of the SimplificationRule class."""

import unittest

import qtoolkit.utils.constants.quantum_gates as qgconsts
import tests.qtestcase as qtest
from qtoolkit.data_structures.quantum_circuit.quantum_circuit import \
    QuantumCircuit
from qtoolkit.data_structures.quantum_circuit.simplifier.gate_parameter import \
    GateParameter
from qtoolkit.data_structures.quantum_circuit.simplifier.simplification_rule \
    import \
    SimplificationRule


class SimplificationRuleTestCase(qtest.QTestCase):
    """Unit-tests for the SimplificationRule class."""

    def setUp(self):
        self._rule_H_H = ['H', 'H']
        self._none_parameters_2 = [None] * 2

        self._non_simplifiable_H_H = QuantumCircuit(1)
        self._non_simplifiable_H_H.apply(qgconsts.H, 0)
        self._non_simplifiable_H_H.apply(qgconsts.X, 0)
        self._non_simplifiable_H_H.apply(qgconsts.X, 0)
        self._non_simplifiable_H_H.apply(qgconsts.H, 0)

        self._simplifiable_H_H = QuantumCircuit(1)
        self._simplifiable_H_H.apply(qgconsts.X, 0)
        self._simplifiable_H_H.apply(qgconsts.H, 0)
        self._simplifiable_H_H.apply(qgconsts.H, 0)
        self._simplifiable_H_H.apply(qgconsts.X, 0)

        self._rule_Rx_Rx = ['Rx', 'Rx']
        self._parameters_inversed_2 = [GateParameter(1, lambda x: x),
                                       GateParameter(1, lambda x: -x)]

    def test_initialisation_simple(self) -> None:
        SimplificationRule(self._rule_H_H, self._none_parameters_2)

    def test_is_simplifiable(self) -> None:
        sr = SimplificationRule(self._rule_H_H, self._none_parameters_2)
        self.assertTrue(
            sr.is_simplifiable(list(self._simplifiable_H_H.gates_on_qubit(0))))
        self.assertFalse(sr.is_simplifiable(
            list(self._non_simplifiable_H_H.gates_on_qubit(0))))


if __name__ == '__main__':
    unittest.main()
