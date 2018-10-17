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

"""Contains a wrapper for nearest-neighbours structures available in Python."""

import typing

import numpy
import scipy.spatial
import sklearn.neighbors

import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq


class NearestNeighbourStructure:
    """A generic and efficient data structure for nearest-neighbour requests.

    This class wraps several data structures that are used for efficient
    nearest-neighbour searches. The wrapped data structures are:
        1. scipy.spatial.cKDTree
        2. sklearn.neighbors.KDTree
        3. sklearn.neighbors.BallTree
    """

    def __init__(self, data: numpy.ndarray, method: str, leaf_size: int,
                 sequences: typing.List[qgate_seq.QuantumGateSequence]) -> None:
        """Initialise the NNStructure instance.

        :param data: An array of doubles with shape (n,m) storing the n data
        points of dimension m to be indexed. This array is **not** copied,
        modifications will result in bogus results.
        :param method: Method used to index the data.
        :param leaf_size: Below this size, a brute-force search is performed.
        :param sequences: pre-computed quantum gate sequences.
        """
        supported_methods = {"cKDTree", "sklKDTree", "sklBallTree"}
        assert method in supported_methods

        self._data = data
        self._sequences = sequences

        if method == "cKDTree":
            self._internal_nn_structure = scipy.spatial.cKDTree(data, leaf_size)
        elif method == "sklKDTree":
            self._internal_nn_structure = sklearn.neighbors.KDTree(data,
                                                                   leaf_size)
        else:
            self._internal_nn_structure = sklearn.neighbors.BallTree(data,
                                                                     leaf_size)

    def query(self, X: numpy.ndarray) -> typing.Tuple[
        float, qgate_seq.QuantumGateSequence]:
        """Query the underlying data structure for nearest-neighbour of X.

        :param X: The point we are searching an approximation for.
        :return: The distance of the found approximation along with the
        corresponding quantum gate sequence.
        """
        if isinstance(self._internal_nn_structure, (
            sklearn.neighbors.kd_tree.KDTree,
            sklearn.neighbors.ball_tree.BallTree)):
            distance, index = self._internal_nn_structure.query(
                X.reshape(1, -1))
            dist, idx = distance[0, 0], index[0, 0]
        else:
            distance, index = self._internal_nn_structure.query(X)
            dist, idx = distance, index

        return dist, self._sequences[idx]
