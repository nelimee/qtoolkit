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

"""Test of the procedures to create random complex numbers."""

import unittest

import numpy

import qtoolkit.maths.random_complex as rand_complex


class RandomComplexTestCase(unittest.TestCase):
    """Unit-tests for the random_complex generation functions."""

    def test_is_complex(self) -> None:
        number = rand_complex.generate_random_complex()
        self.assertTrue(isinstance(number, complex))

    def test_complex_amplitude(self) -> None:
        for amplitude in numpy.linspace(0, 100, 1000):
            self.assertAlmostEqual(amplitude, numpy.abs(
                rand_complex.generate_random_complex(amplitude)))

    def test_complex_vector_size(self) -> None:
        for size in (0, 1, 100, 10000):
            self.assertEqual(size,
                             rand_complex.generate_random_normalised_complexes(
                                 size).size)

    def test_complex_vector_negative_size(self) -> None:
        with self.assertRaises(ValueError):
            rand_complex.generate_random_normalised_complexes(-1)

    def test_complex_vector_normalised(self) -> None:
        for size in (1, 100, 10000):
            self.assertAlmostEqual(1.0, numpy.linalg.norm(
                rand_complex.generate_random_normalised_complexes(size)))


if __name__ == '__main__':
    unittest.main()
