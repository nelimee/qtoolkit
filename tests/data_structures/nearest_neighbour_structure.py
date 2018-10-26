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

"""Test of the nearest-neighbour structure."""

import unittest

import numpy

import qtoolkit.data_structures.nearest_neighbour_structure as nn_structure
import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq
import qtoolkit.maths.matrix.generation.su2 as gen_su2
import qtoolkit.utils.constants.matrices as mconsts
import tests.qtestcase as qtest


class NearestNeighbourStructureTestCase(qtest.QTestCase):
    """Unit-tests for the nearest-neighbour structure."""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup done once for all. Not repeated at each test.

        This setup is done only once because the unit-tests only need
        read-only data and the cost of constructing everything before
        each test may be non-negligible.
        """
        dataset_size = 1000
        dimension = 3

        cls._basis = [mconsts.H_SU2, mconsts.T_SU2, mconsts.T_SU2.T.conj()]
        cls._sequences = [qgate_seq.QuantumGateSequence(cls._basis,
                                                        numpy.random.randint(0,
                                                                             len(
                                                                                 cls._basis),
                                                                             5))
                          for _ in range(dataset_size)]
        cls._data = numpy.array([seq.matrix for seq in cls._sequences])
        cls._int_sequences = numpy.array([seq.gates for seq in cls._sequences])

    def test_construction(self) -> None:
        """Tests the construction ."""
        nn_structure.NearestNeighbourStructure(self._data, self._int_sequences,
                                               self._basis)

    def _random_query_su2(self,
                          nn_struct: nn_structure.NearestNeighbourStructure) \
        -> None:
        """Perform a random query and check the validity of the result."""
        random_matrix = gen_su2.generate_random_SU2_matrix()
        dist, res = nn_struct.query(random_matrix)
        error = numpy.linalg.norm(random_matrix - res.matrix)
        self.assertAlmostEqual(error, dist)

    def test_query_su2(self) -> None:
        """Tests query validity."""
        nn_struct = nn_structure.NearestNeighbourStructure(self._data,
                                                           self._int_sequences,
                                                           self._basis)
        self._random_query_su2(nn_struct)


if __name__ == '__main__':
    unittest.main()
