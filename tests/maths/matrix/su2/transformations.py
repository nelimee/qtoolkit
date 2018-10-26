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

"""Tests for the transformation.py file."""

import typing
import unittest

import numpy
import scipy

import qtoolkit.maths.matrix.generation.su2 as gen_su2
import qtoolkit.maths.matrix.generation.unitary as gen_unitary
import qtoolkit.maths.matrix.su2.transformations as trans
import qtoolkit.maths.random as rand
import qtoolkit.utils.constants.matrices as mconsts
import qtoolkit.utils.constants.others as other_consts
import tests.qtestcase as qtest


class TransformationsSU2SO3TestCase(qtest.QTestCase):
    """Tests for the transformations between SO(3) and SU(2)."""

    def test_su2_to_so3_identity(self) -> None:
        """Tests su2_to_so3 for multiples of the identity matrix."""
        expected_coefficients = numpy.array([2 * numpy.pi, 0.0, 0.0])
        for coefficient in numpy.linspace(-1, 1, 100):
            result = trans.su2_to_so3(coefficient * mconsts.IDENTITY_2X2)
            self.assert2NormClose(expected_coefficients, result)

    def test_su2_to_so3_pauli_x(self) -> None:
        """Tests su2_to_so3 for the Pauli X matrix."""
        expected_coefficients = numpy.array([1.0, 0.0, 0.0])
        result = trans.su2_to_so3(scipy.linalg.expm(-1.j * mconsts.P_X / 2))
        self.assert2NormClose(expected_coefficients, result)

    def test_su2_to_so3_pauli_y(self) -> None:
        """Tests su2_to_so3 for the Pauli Y matrix."""
        expected_coefficients = numpy.array([0.0, 1.0, 0.0])
        result = trans.su2_to_so3(scipy.linalg.expm(-1.j * mconsts.P_Y / 2))
        self.assert2NormClose(expected_coefficients, result)

    def test_su2_to_so3_pauli_z(self) -> None:
        """Tests su2_to_so3 for the Pauli Z matrix."""
        expected_coefficients = numpy.array([0.0, 0.0, 1.0])
        result = trans.su2_to_so3(scipy.linalg.expm(-1.j * mconsts.P_Z / 2))
        self.assert2NormClose(expected_coefficients, result)

    def test_so3_to_su2_all_zero(self) -> None:
        """Tests so3_to_su2 for a zero SO(3) vector."""
        matrix = trans.so3_to_su2(numpy.array([0.0, 0.0, 0.0]))
        self.assert2NormClose(matrix, matrix[0, 0] * numpy.identity(2))

    def test_so3_to_su2_pauli_x(self) -> None:
        """Tests so3_to_su2 for a vector representing the Pauli X matrix."""
        matrix = trans.so3_to_su2(numpy.array([1.0, 0.0, 0.0]))
        self.assert2NormClose(matrix, scipy.linalg.expm(-1.j * mconsts.P_X / 2))

    def test_so3_to_su2_pauli_y(self) -> None:
        """Tests so3_to_su2 for a vector representing the Pauli Y matrix."""
        matrix = trans.so3_to_su2(numpy.array([0.0, 1.0, 0.0]))
        self.assert2NormClose(matrix, scipy.linalg.expm(-1.j * mconsts.P_Y / 2))

    def test_so3_to_su2_pauli_z(self) -> None:
        """Tests so3_to_su2 for a vector representing the Pauli Z matrix."""
        matrix = trans.so3_to_su2(numpy.array([0.0, 0.0, 1.0]))
        self.assert2NormClose(matrix, scipy.linalg.expm(-1.j * mconsts.P_Z / 2))

    def test_so3_to_su2_to_so3_random(self) -> None:
        """Tests X == su2_to_so3(so3_to_su2(X)) for random X."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return
        for idx in range(other_consts.RANDOM_SAMPLES):
            local_coefficients = numpy.random.rand(3)
            self.assert2NormClose(
                trans.su2_to_so3(trans.so3_to_su2(local_coefficients)),
                local_coefficients)

    def test_su2_to_so3_to_su2_random(self) -> None:
        """Tests X == so3_to_su2(su2_to_so3(X)) for random X."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return
        for idx in range(other_consts.RANDOM_SAMPLES):
            local_su2 = gen_su2.generate_random_SU2_matrix()
            self.assert2NormClose(trans.so3_to_su2(trans.su2_to_so3(local_su2)),
                                  local_su2)


class TransformationsSU2HTestCase(qtest.QTestCase):
    """Tests for the transformations between SU(2) and H (quaternions)."""

    def test_H_to_su2_identity(self) -> None:
        """Tests H_to_su2 for the identity matrix."""
        mat = trans.H_to_su2(numpy.array([1, 0, 0, 0]))
        self.assertOperatorNormClose(mat, mconsts.ID2_SU2)

    def test_H_to_su2_pauli_x(self) -> None:
        """Tests H_to_su2 for the Pauli X matrix."""
        mat = trans.H_to_su2(numpy.array([0, 1, 0, 0]))
        self.assertOperatorNormClose(mat, mconsts.P_X_SU2)

    def test_H_to_su2_pauli_y(self) -> None:
        """Tests H_to_su2 for the Pauli Y matrix."""
        mat = trans.H_to_su2(numpy.array([0, 0, 1, 0]))
        self.assertOperatorNormClose(mat, mconsts.P_Y_SU2)

    def test_H_to_su2_pauli_z(self) -> None:
        """Tests H_to_su2 for the Pauli Z matrix."""
        mat = trans.H_to_su2(numpy.array([0, 0, 0, 1]))
        self.assertOperatorNormClose(mat, mconsts.P_Z_SU2)

    def test_su2_to_H_identity(self) -> None:
        """Tests su2_to_H for the identity matrix."""
        quaternion = trans.su2_to_H(mconsts.ID2_SU2)
        self.assert2NormClose(quaternion, numpy.array([1, 0, 0, 0]))

    def test_su2_to_H_pauli_x(self) -> None:
        """Tests su2_to_H for the Pauli X matrix."""
        quaternion = trans.su2_to_H(mconsts.P_X_SU2)
        self.assert2NormClose(quaternion, numpy.array([0, 1, 0, 0]))

    def test_su2_to_H_pauli_y(self) -> None:
        """Tests su2_to_H for the Pauli Y matrix."""
        quaternion = trans.su2_to_H(mconsts.P_Y_SU2)
        self.assert2NormClose(quaternion, numpy.array([0, 0, 1, 0]))

    def test_su2_to_H_pauli_z(self) -> None:
        """Tests su2_to_H for the Pauli Z matrix."""
        quaternion = trans.su2_to_H(mconsts.P_Z_SU2)
        self.assert2NormClose(quaternion, numpy.array([0, 0, 0, 1]))

    def test_su2_to_H_to_su2_random(self) -> None:
        """Tests X == H_to_su2(su2_to_H(X)) for random X."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return
        for idx in range(other_consts.RANDOM_SAMPLES):
            local_su2 = gen_su2.generate_random_SU2_matrix()
            self.assertOperatorNormClose(
                trans.H_to_su2(trans.su2_to_H(local_su2)), local_su2)

    def test_H_to_su2_to_H_random(self) -> None:
        """Tests X == su2_to_H(H_to_su2(X)) for random X."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return
        for idx in range(other_consts.RANDOM_SAMPLES):
            quaternion = rand.generate_unit_norm_quaternion()
            self.assert2NormClose(trans.su2_to_H(trans.H_to_su2(quaternion)),
                                  quaternion)


class TransformationsSU2UnitTestCase(qtest.QTestCase):
    """Tests for the transformations between SU(2) and unitary matrices."""

    def test_unitary_to_su2_identity(self) -> None:
        """Test validity of unitary_to_su2 for the identity matrix."""
        self.assertOperatorNormClose(mconsts.ID2,
                                     trans.unitary_to_su2(mconsts.ID2))

    def test_unitary_to_su2_pauli_x(self) -> None:
        """Test validity of unitary_to_su2 for the Pauli X matrix."""
        self.assertOperatorNormClose(mconsts.P_X_SU2,
                                     trans.unitary_to_su2(mconsts.P_X))

    def test_unitary_to_su2_pauli_y(self) -> None:
        """Test validity of unitary_to_su2 for the Pauli Y matrix."""
        self.assertOperatorNormClose(mconsts.P_Y_SU2,
                                     trans.unitary_to_su2(mconsts.P_Y))

    def test_unitary_to_su2_pauli_z(self) -> None:
        """Test validity of unitary_to_su2 for the Pauli Z matrix."""
        self.assertOperatorNormClose(mconsts.P_Z_SU2,
                                     trans.unitary_to_su2(mconsts.P_Z))

    def test_unitary_to_su2_repeated_random(self) -> None:
        """Test that repetition of unitary_to_su2 has no effect."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return

        ReturnType = typing.TypeVar("ReturnType")

        def repeat_func(f: typing.Callable[[ReturnType], ReturnType], n: int,
                        args: ReturnType) -> ReturnType:
            for i in range(n):
                args = f(args)
            return args

        for idx in range(other_consts.RANDOM_SAMPLES):
            unitary = gen_unitary.generate_random_unitary_matrix()
            # We will repeat the application of unitary_to_su2 up to 6 times.
            # 6 has been arbitrarily chosen.
            su2_matrix = trans.unitary_to_su2(unitary)
            for i in range(6):
                self.assertOperatorNormClose(
                    repeat_func(trans.unitary_to_su2, i, su2_matrix),
                    su2_matrix)

    def test_unitary_to_su2_random(self) -> None:
        """Tests det(unitary_to_su2(X)) == 1 for random unitary X."""
        # Abort if we don't want random tests
        if not other_consts.USE_RANDOM_TESTS:
            return
        for idx in range(other_consts.RANDOM_SAMPLES):
            unitary = gen_unitary.generate_random_unitary_matrix()
            self.assertAlmostEqual(
                numpy.linalg.det(trans.unitary_to_su2(unitary)), 1.0)


if __name__ == '__main__':
    unittest.main()
