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

"""Test of the procedures used to generate SU(2) matrices."""

import unittest

import qtoolkit.maths.matrix.generation.su2 as gen_su2
import qtoolkit.utils.constants.others as other_consts
import tests.qtestcase as qtest


class Su2TestCase(qtest.QTestCase):
    """Unit-test for the SU(2) generation functions."""

    def test_random_su2_matrix(self) -> None:
        """Tests if the matrices obtained by generate_random_SU2_matrix
        are in SU(2)."""
        if other_consts.USE_RANDOM_TESTS:
            for _ in range(other_consts.RANDOM_SAMPLES):
                M = gen_su2.generate_random_SU2_matrix()
                self.assertSU2Matrix(M)


if __name__ == '__main__':
    unittest.main()
