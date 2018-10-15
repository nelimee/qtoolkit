# ======================================================================
# Copyright CERFACS (October 2018)
# Contributor: Adrien Suau (suau@cerfacs.fr)
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

"""Base class for all the tests of qtoolkit module."""

import unittest

import numpy

import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.utils.constants as qconsts
import qtoolkit.utils.types as qtypes


class QTestCase(unittest.TestCase):
    """Base class for all the tests of qtoolkit module."""

    def assert2NormClose(self, array1, array2, rtol: float = 1e-5) -> None:
        """Check if the two arrays are close to each other.

        The check is performed by computing the 2-norm of their
        element-wise difference and check that this norm is below a
        given tolerence.

        :param array1: First array.
        :param array2: Second array.
        :param rtol: Relative tolerance. See numpy.isclose documentation
        for a more detailed explanation.
        """
        message = (f"Arrays a1 = \n{array1}\nand a2 = \n{array2}\nare not "
                   f"close enough! ||a1-a2||_2 = "
                   f"{numpy.linalg.norm(array1 - array2, 2)}.")
        self.assertTrue(
            numpy.isclose(numpy.linalg.norm(array1 - array2), 0, rtol=rtol,
                          atol=1e-7), msg=message)

    def assertOperatorNormClose(self, U: qtypes.UnitaryMatrix,
                                V: qtypes.UnitaryMatrix, rtol: float = 1e-5,
                                atol: float = 1e-8) -> None:
        """Check if the two unitary matrices are close to each other.

        The check is performed by computing the operator norm of the
        difference of the two matrices and check that this norm is below
        a given tolerence.

        :param U: First array.
        :param V: Second array.
        :param rtol: Relative tolerance. See numpy.isclose documentation
        for a more detailed explanation.
        :param atol: Absolute tolerance. See numpy.isclose documentation
        for a more detailed explanation.
        """
        message = (f"Matrices U = \n{U}\nand V = \n{V}\nare not close "
                   f"enough! ||U-V|| = {qdists.operator_norm(U - V)}.")
        self.assertTrue(
            numpy.isclose(qdists.operator_norm(U - V), 0, rtol=rtol, atol=atol),
            msg=message)

    def assertSU2Matrix(self, M: qtypes.GenericMatrix) -> None:
        """Check if the given matrix is in SU(2).

        :param M: The matrix to check.
        """
        self.assertUnitaryMatrix(M)
        self.assertAlmostEqual(numpy.linalg.det(M), 1.0)

    def assertUnitaryMatrix(self, M: qtypes.GenericMatrix) -> None:
        """Check if the given matrix is unitary.

        :param M: The matrix to check.
        """
        self.assert2NormClose(qconsts.ID2_SU2, M @ M.T.conj())
