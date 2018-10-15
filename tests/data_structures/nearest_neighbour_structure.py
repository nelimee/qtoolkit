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

import unittest

import numpy

import qtoolkit.data_structures.nearest_neighbour_structure as NNstruct
import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq
import qtoolkit.maths.matrix.generation.su2 as gen_su2
import qtoolkit.maths.matrix.su2.transformations as su2_trans
import qtoolkit.utils.constants as qconsts
import tests.qtestcase as qtest


class NearestNeighbourStructureTestCase(qtest.QTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dataset_size = 1000
        dimension = 3

        cls._basis = [qconsts.H_SU2, qconsts.T_SU2, qconsts.T_SU2.T.conj()]
        cls._sequences = [qgate_seq.QuantumGateSequence(cls._basis,
                                                        numpy.random.randint(0,
                                                                             len(
                                                                                 cls._basis),
                                                                             5))
                          for _ in range(dataset_size)]
        cls._data = numpy.array(
            [su2_trans.su2_to_so3(seq.matrix) for seq in cls._sequences])

    def test_scipy_cKDTree_construction(self) -> None:
        NNstruct.NearestNeighbourStructure(self._data, 'cKDTree', 40,
                                           self._sequences)

    def test_sklearn_KDTree_construction(self) -> None:
        NNstruct.NearestNeighbourStructure(self._data, 'sklKDTree', 40,
                                           self._sequences)

    def test_sklearn_BallTree_construction(self) -> None:
        NNstruct.NearestNeighbourStructure(self._data, 'sklBallTree', 40,
                                           self._sequences)

    def _random_query(self,
                      nn_struct: NNstruct.NearestNeighbourStructure) -> None:
        random_matrix = gen_su2.generate_random_SU2_matrix()
        query = su2_trans.su2_to_so3(random_matrix)
        dist, res = nn_struct.query(query)
        error = numpy.linalg.norm(query - su2_trans.su2_to_so3(res.matrix))
        self.assertAlmostEqual(error, dist)

    def test_scipy_cKDTree_query(self) -> None:
        nn_struct = NNstruct.NearestNeighbourStructure(self._data, 'cKDTree',
                                                       40, self._sequences)
        self._random_query(nn_struct)

    def test_sklearn_KDTree_query(self) -> None:
        nn_struct = NNstruct.NearestNeighbourStructure(self._data, 'sklKDTree',
                                                       40, self._sequences)
        self._random_query(nn_struct)

    def test_sklearn_BallTree_query(self) -> None:
        nn_struct = NNstruct.NearestNeighbourStructure(self._data,
                                                       'sklBallTree', 40,
                                                       self._sequences)
        self._random_query(nn_struct)


if __name__ == '__main__':
    unittest.main()
