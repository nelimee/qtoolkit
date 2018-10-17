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

"""Test of the quantum gate sequence structure."""

import unittest

import numpy

import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq
import qtoolkit.utils.constants as qconsts
import tests.qtestcase as qtest


class QuantumGateSequenceTestCase(qtest.QTestCase):
    """Unit-tests for the QuantumGateSequence class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compute and store constants re-used during the tests."""
        cls._basis_SU2_HT = [qconsts.H_SU2, qconsts.H_SU2.T.conj(),
                             qconsts.T_SU2, qconsts.T_SU2.T.conj()]
        cls._basis_SU2_HT_inverses = numpy.array([1, 0, 3, 2])
        cls._basis_SU2_HTS = [qconsts.H_SU2, qconsts.H_SU2.T.conj(),
                              qconsts.T_SU2, qconsts.T_SU2.T.conj(),
                              qconsts.S_SU2, qconsts.S_SU2.T.conj()]
        cls._basis_SU2_HTS_inverses = numpy.array([1, 0, 3, 2, 5, 4])

        cls._small_gate_sequence = numpy.random.randint(0,
                                                        len(cls._basis_SU2_HT),
                                                        3)
        cls._small_resulting_matrix = cls._basis_SU2_HT[
                                          cls._small_gate_sequence[0]] @ \
                                      cls._basis_SU2_HT[
                                          cls._small_gate_sequence[1]] @ \
                                      cls._basis_SU2_HT[
                                          cls._small_gate_sequence[2]]
        cls._huge_gate_sequence = numpy.random.randint(0,
                                                       len(cls._basis_SU2_HT),
                                                       1000)

    def test_construction(self) -> None:
        """Tests if the construction works with simple parameters."""
        qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                      self._small_gate_sequence)

    def test_construction_long_gate_sequence(self) -> None:
        """Tests if the construction works with a long gate sequence."""
        qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                      self._huge_gate_sequence)

    def test_construction_with_inverses(self) -> None:
        """Tests if the construction works when inverses are provided."""
        qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                      self._small_gate_sequence,
                                      inverses=self._basis_SU2_HT_inverses)

    def test_construction_with_matrix(self) -> None:
        """Tests if the construction works when the matrix is provided."""
        qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                      self._small_gate_sequence,
                                      resulting_matrix=(
                                          self._small_resulting_matrix))

    def test_construction_with_matrix_and_inverse(self) -> None:
        """Tests __init__ when the matrix and the inverses are provided."""
        qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                      self._small_gate_sequence,
                                      resulting_matrix=(
                                          self._small_resulting_matrix),
                                      inverses=self._basis_SU2_HT_inverses)

    def test_matrix_computation_when_provided_at_construction(self) -> None:
        """Tests if the matrix computation works when the matrix is provided."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence,
                                              resulting_matrix=(
                                                  self._small_resulting_matrix))
        self.assertAllClose(qgate.matrix, self._small_resulting_matrix)

    def test_matrix_computation_when_not_provided_at_construction(self) -> None:
        """Tests the matrix computation when the matrix is not provided."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence)
        self.assertAllClose(qgate.matrix, self._small_resulting_matrix)

    def test_inverse_computation_when_provided_at_construction(self) -> None:
        """Tests if the inverses computation works when they are provided."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence,
                                              self._basis_SU2_HT_inverses)
        self.assertAllEqual(self._basis_SU2_HT_inverses, qgate.inverses)

    def test_inverse_computation_when_not_provided_at_construction(
        self) -> None:
        """Tests if the inverses computation works when they are not
        provided."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence)
        self.assertAllEqual(self._basis_SU2_HT_inverses, qgate.inverses)

    def test_matmul(self) -> None:
        """Tests the __matmul__ implementation."""
        other_gate_sequence = numpy.random.randint(0, len(self._basis_SU2_HT),
                                                   3)
        self_qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                                   self._small_gate_sequence)
        other_qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                                    other_gate_sequence)
        # Calling __matmul__
        res_qgate = self_qgate @ other_qgate
        self.assertAllClose(res_qgate.matrix,
                            self_qgate.matrix @ other_qgate.matrix)
        self.assertAllEqual(res_qgate.gates, numpy.concatenate(
            (self_qgate.gates, other_qgate.gates)))

    def test_inverse(self) -> None:
        """Tests the inverse() implementation."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence)
        inv = qgate.inverse()
        IDENTITY = numpy.identity(qgate.dimension)
        BASIS = self._basis_SU2_HT
        self.assertAllClose(IDENTITY, qgate.matrix @ inv.matrix)
        for gate, inv_gate in zip(qgate.gates, reversed(inv.gates)):
            self.assertAllClose(IDENTITY, BASIS[gate] @ BASIS[inv_gate])

    def test_inverse_with_incomplete_basis(self) -> None:
        """Tests the inverse() method when the basis is not complete."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT[1:],
                                              self._small_gate_sequence)
        with self.assertRaises(RuntimeError):
            qgate.inverse()

    def test_inverse_twice(self) -> None:
        """Tests the inverse() method when called twice on the same instance."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence)
        inv1 = qgate.inverse()
        inv2 = qgate.inverse()
        self.assertAllClose(inv1.matrix, inv2.matrix)
        self.assertAllEqual(inv1.gates, inv2.gates)

    def test_inverse_when_matrix_computed(self) -> None:
        """Tests the inverse() method when the matrix is already computed."""
        qgate = qgate_seq.QuantumGateSequence(self._basis_SU2_HT,
                                              self._small_gate_sequence,
                                              resulting_matrix=(
                                                  self._small_resulting_matrix))
        inv = qgate.inverse()
        self.assertAllClose(inv.matrix,
                            numpy.linalg.inv(self._small_resulting_matrix))


if __name__ == '__main__':
    unittest.main()
