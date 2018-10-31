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

import numpy

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.utils.constants.matrices as mconsts
import qtoolkit.utils.constants.quantum_gates as qgconsts
import tests.qtestcase as qtest


class QuantumCircuitTestCase(qtest.QTestCase):
    """Unit-tests for the QuantumCircuit class."""

    def _chained_left_kron(self, *matrices: numpy.ndarray) -> numpy.ndarray:
        if len(matrices) == 1: return matrices[0]
        return numpy.kron(matrices[0], self._chained_left_kron(*matrices[1:]))

    def setUp(self):
        self._circ1 = qcirc.QuantumCircuit(1)
        self._circ3 = qcirc.QuantumCircuit(3)

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
        self.assertEqual(self._circ3.size, 3)

    def test_apply_simple(self) -> None:
        self._circ1.apply(qgconsts.X, 0)
        operations = list(self._circ1.operations)
        self.assertEqual(len(operations), 1)
        self.assertAllClose(operations[0].gate.matrix, qgconsts.X.matrix)

    def test_apply_CX_simple(self) -> None:
        self._circ3.apply(qgconsts.X, 2, [1])
        operations = list(self._circ3.operations)
        self.assertEqual(len(operations), 1)
        matrix = self._chained_left_kron(mconsts.ID2, mconsts.P0,
                                         mconsts.ID2) + self._chained_left_kron(
            mconsts.ID2, mconsts.P1, mconsts.X)
        self.assertAllClose(self._circ3.matrix, matrix)

    def test_apply_CX_reversed(self) -> None:
        self._circ3.apply(qgconsts.X, 1, [2])
        operations = list(self._circ3.operations)
        self.assertEqual(len(operations), 1)
        matrix = self._chained_left_kron(mconsts.ID2, mconsts.ID2,
                                         mconsts.P0) + self._chained_left_kron(
            mconsts.ID2, mconsts.X, mconsts.P1)
        self.assertAllClose(self._circ3.matrix, matrix)

    def test_apply_CX_not_neighbour(self) -> None:
        self._circ3.apply(qgconsts.X, 0, [2])
        operations = list(self._circ3.operations)
        self.assertEqual(len(operations), 1)
        matrix = self._chained_left_kron(mconsts.ID2, mconsts.ID2,
                                         mconsts.P0) + self._chained_left_kron(
            mconsts.X, mconsts.ID2, mconsts.P1)
        self.assertAllClose(self._circ3.matrix, matrix)

    def test_apply_target_out_of_range_qubit(self) -> None:
        with self.assertRaises(IndexError):
            self._circ1.apply(qgconsts.X, 1)
        with self.assertRaises(IndexError):
            self._circ1.apply(qgconsts.X, -1)
        with self.assertRaises(IndexError):
            self._circ3.apply(qgconsts.X, 4)

    def test_apply_control_out_of_range_qubit(self) -> None:
        with self.assertRaises(IndexError):
            self._circ1.apply(qgconsts.X, 0, [1])
        with self.assertRaises(IndexError):
            self._circ1.apply(qgconsts.X, 0, [-1])
        with self.assertRaises(IndexError):
            self._circ3.apply(qgconsts.X, 1, [4])

    def test_remove_last_inserted_simple(self) -> None:
        self._circ1.apply(qgconsts.X, 0)
        self._circ1.pop()
        self.assertFalse(list(self._circ1.operations),
                         "There should be no quantum gate applied to qubit "
                         "n°0.")

    def test_remove_last_inserted_multiple_inserted(self) -> None:
        self._circ1.apply(qgconsts.X, 0)
        self._circ1.apply(qgconsts.S, 0)
        self._circ1.apply(qgconsts.H, 0)
        self._circ1.pop()
        self.assertEqual(len(list(self._circ1.operations)), 2)

    def test_remove_last_inserted_multiple_calls(self) -> None:
        self._circ1.apply(qgconsts.X, 0)
        self._circ1.apply(qgconsts.S, 0)
        self._circ1.apply(qgconsts.H, 0)
        self._circ1.pop()
        self._circ1.pop()
        self.assertEqual(len(list(self._circ1.operations)), 1)

    def test_remove_last_inserted_on_empty_circuit(self) -> None:
        with self.assertRaises(RuntimeError):
            self._circ1.pop()

    def test_get_last_inserted_qubits_simple(self) -> None:
        self._circ1.apply(qgconsts.X, 0)
        qop = self._circ1.pop()
        self.assertEqual(qop.target, 0)
        self.assertFalse(qop.controls, "There should be no control qubit.")
        self.assertEqual(qgconsts.X.name, qop.gate.name)
        self.assertAllClose(qgconsts.X.matrix, qop.gate.matrix)

    def test_get_last_inserted_qubit_CX(self) -> None:
        self._circ3.apply(qgconsts.X, 1, [2])
        qop = self._circ3.pop()
        self.assertEqual(qop.target, 1)
        self.assertEqual(len(qop.controls), 1)
        self.assertEqual(qop.controls[0], 2)


if __name__ == '__main__':
    unittest.main()
