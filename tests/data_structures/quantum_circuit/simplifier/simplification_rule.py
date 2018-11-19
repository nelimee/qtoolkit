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

import qtoolkit.data_structures.quantum_circuit.simplifier.simplification_rule \
    as sr
import qtoolkit.utils.constants.quantum_gates as qgconsts
import tests.qtestcase as qtest
from qtoolkit.data_structures.quantum_circuit.quantum_circuit import \
    QuantumCircuit


class SimplificationRuleTestCase(qtest.QTestCase):
    """Unit-tests for the SimplificationRule class."""

    def test_initialisation_simple_inverse_rule(self) -> None:
        sr.InverseRule(qgconsts.H)
        sr.InverseRule(qgconsts.T)
        sr.InverseRule(qgconsts.X)
        sr.InverseRule(qgconsts.Y)
        sr.InverseRule(qgconsts.Z)
        sr.InverseRule(qgconsts.S)

    def test_initialisation_simple_inverse_rule_inverse(self) -> None:
        sr.InverseRule(qgconsts.H.H)
        sr.InverseRule(qgconsts.T.H)
        sr.InverseRule(qgconsts.X.H)
        sr.InverseRule(qgconsts.Y.H)
        sr.InverseRule(qgconsts.Z.H)
        sr.InverseRule(qgconsts.S.H)

    def test_initialisation_simple_CX_rule(self) -> None:
        sr.CXInverseRule()

    def test_is_simplifiable_from_last_inverse_rule_empty_circuit(self) -> None:
        simpl_HH = sr.InverseRule(qgconsts.H)
        empty_circuit = QuantumCircuit(1)
        self.assertFalse(simpl_HH.is_simplifiable_from_last(empty_circuit))

    def test_is_simplifiable_from_last_inverse_rule(self) -> None:
        # Rule creation
        simpl_HH = sr.InverseRule(qgconsts.H)
        simpl_XX = sr.InverseRule(qgconsts.X)

        quantum_circuit = QuantumCircuit(1)
        quantum_circuit.apply(qgconsts.X, 0)
        self.assertFalse(simpl_HH.is_simplifiable_from_last(quantum_circuit))
        self.assertFalse(simpl_XX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.H, 0)
        self.assertFalse(simpl_HH.is_simplifiable_from_last(quantum_circuit))
        self.assertFalse(simpl_XX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.H, 0)
        self.assertTrue(simpl_HH.is_simplifiable_from_last(quantum_circuit))
        self.assertFalse(simpl_XX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0)
        self.assertFalse(simpl_HH.is_simplifiable_from_last(quantum_circuit))
        self.assertFalse(simpl_XX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0)
        self.assertFalse(simpl_HH.is_simplifiable_from_last(quantum_circuit))
        self.assertTrue(simpl_XX.is_simplifiable_from_last(quantum_circuit))

    def test_is_simplifiable_CX_inverse_rule(self) -> None:
        # Rule creation
        simpl_CX = sr.CXInverseRule()

        quantum_circuit = QuantumCircuit(2)
        quantum_circuit.apply(qgconsts.X, 0)
        self.assertFalse(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0, [1])
        self.assertFalse(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0)
        self.assertFalse(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0, [1])
        self.assertFalse(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0, [1])
        self.assertTrue(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 0, [1])
        self.assertTrue(simpl_CX.is_simplifiable_from_last(quantum_circuit))
        quantum_circuit.apply(qgconsts.X, 1, [0])
        self.assertFalse(simpl_CX.is_simplifiable_from_last(quantum_circuit))


if __name__ == '__main__':
    unittest.main()
