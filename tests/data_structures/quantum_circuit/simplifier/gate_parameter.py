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

"""Test of the GateParameter class."""

import unittest

import tests.qtestcase as qtest
from qtoolkit.data_structures.quantum_circuit.simplifier.gate_parameter import \
    GateParameter


class GateParameterTestCase(qtest.QTestCase):
    """Unit-tests for the GateParameter class."""

    def test_initialisation_with_integer(self) -> None:
        GateParameter(0)
        GateParameter(1)
        GateParameter(-1)

    def test_initialisation_with_string(self) -> None:
        GateParameter("a")
        GateParameter("a_not_so_empty_name_repeated" * 100)

    def test_default_initialisation_lambda(self) -> None:
        gp = GateParameter(1)
        for i in range(10):
            self.assertEqual(gp.apply_transformation(i), i)

    def test_apply_transformation_with_lambda(self) -> None:
        gp = GateParameter(1, lambda x: x + 1)
        for i in range(10):
            self.assertEqual(gp.apply_transformation(i), i + 1)

    def test_apply_transformation_with_function_def(self) -> None:
        def f(x): return 2 * x + 7

        gp = GateParameter(1, f)
        for i in range(10):
            self.assertEqual(gp.apply_transformation(i), f(i))


if __name__ == '__main__':
    unittest.main()
