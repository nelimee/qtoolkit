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

"""Test of the QuantumCircuit class."""

import unittest

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.utils.constants.quantum_gates as qgconsts
import tests.qtestcase as qtest


class QuantumCircuitTestCase(qtest.QTestCase):
    """Unit-tests for the QuantumCircuit class."""

    def setUp(self):
        self._circ1 = qcirc.QuantumCircuit(1)
        self._circ10 = qcirc.QuantumCircuit(10)

    def test_construction_negative_qubit_number(self) -> None:
        with self.assertRaises(AssertionError):
            qcirc.QuantumCircuit(-1)
        with self.assertRaises(AssertionError):
            qcirc.QuantumCircuit(0)

    def test_construction_positive_qubit_number(self) -> None:
        qcirc.QuantumCircuit(1)
        qcirc.QuantumCircuit(10)

    def test_size(self) -> None:
        self.assertEqual(self._circ1.size, 1)
        self.assertEqual(self._circ10.size, 10)

    def test_apply_simple(self) -> None:
        self._circ1.apply(qgconsts.X, [0])
        self.assertAllClose(self._circ1.qubits[0][0].matrix, qgconsts.X.matrix)

    def test_apply_CX(self) -> None:
        self._circ10.apply(qgconsts.CX, [1, 2])
        self.assertEqual(self._circ10.qubits[1][0].name, qgconsts.CX_ctrl.name)
        self.assertEqual(self._circ10.qubits[2][0].name, qgconsts.CX_trgt.name)

    def test_apply_too_much_qubits(self) -> None:
        with self.assertRaises(AssertionError):
            self._circ10.apply(qgconsts.X, [0, 1, 2])
        with self.assertRaises(AssertionError):
            self._circ10.apply(qgconsts.X, [0, 1])

    def test_apply_not_enough_qubits(self) -> None:
        with self.assertRaises(AssertionError):
            self._circ10.apply(qgconsts.CX, [6])
        with self.assertRaises(AssertionError):
            self._circ1.apply(qgconsts.X, [])

    def test_apply_out_of_range_qubit(self) -> None:
        with self.assertRaises(IndexError):
            self._circ1.apply(qgconsts.X, [1])

    def test_remove_last_inserted_simple(self) -> None:
        self._circ1.apply(qgconsts.X, [0])
        self._circ1.remove_last_inserted()
        self.assertFalse(self._circ1.qubits[0],
                         "There should be no quantum gate applied to qubit "
                         "n°0.")

    def test_remove_last_inserted_multiple_inserted(self) -> None:
        self._circ1.apply(qgconsts.X, [0])
        self._circ1.apply(qgconsts.S, [0])
        self._circ1.apply(qgconsts.H, [0])
        self._circ1.remove_last_inserted()
        self.assertEqual(len(self._circ1.qubits[0]), 2)

    def test_remove_last_inserted_multiple_calls(self) -> None:
        self._circ1.apply(qgconsts.X, [0])
        self._circ1.apply(qgconsts.S, [0])
        self._circ1.apply(qgconsts.H, [0])
        self._circ1.remove_last_inserted()
        self._circ1.remove_last_inserted()
        self.assertEqual(len(self._circ1.qubits[0]), 1)

    def test_remove_last_inserted_on_empty_circuit(self) -> None:
        with self.assertRaises(RuntimeError):
            self._circ1.remove_last_inserted()

    def test_get_last_inserted_qubits_simple(self) -> None:
        self._circ1.apply(qgconsts.X, [0])
        instructions = self._circ1.get_last_modified_qubits()
        self.assertEqual(len(instructions), 1)
        self.assertEqual(instructions[0][0].name, qgconsts.X.name)
        self.assertAllClose(instructions[0][0].matrix, qgconsts.X.matrix)

    def test_get_last_inserted_qubit_CX(self) -> None:
        self._circ10.apply(qgconsts.CX, [0, 1])
        instructions = self._circ10.get_last_modified_qubits()
        self.assertEqual(len(instructions), 2)
        self.assertEqual(instructions[0][0].name, qgconsts.CX_ctrl.name)
        self.assertEqual(instructions[1][0].name, qgconsts.CX_trgt.name)

    def test_get_last_inserted_qubit_empty(self) -> None:
        with self.assertRaises(RuntimeError):
            self._circ1.get_last_modified_qubits()


if __name__ == '__main__':
    unittest.main()
