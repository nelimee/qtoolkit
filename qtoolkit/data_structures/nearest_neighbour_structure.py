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

"""Contains a wrapper on nearest-neighbours structures available in Python.

Currently, the wrapper is quite inefficient because it initialise and query
sequentially 2 different data-structures:

#. An instance of :py:class:`annoy.Annoy` specialised in performing
   **approximate** nearest-neighbour (ANN) queries.
#. An instance of :py:class:`scipy.spatial.cKDTree` specialised in
   performing **exact** nearest-neighbour (ENN or NN) queries.

For the moment the 2 data-structures are kept like this. The reason is that I
need more insights on several factors to know which one I should use:

#. The maximal number of qubits on which the Solovay-Kitaev algorithm will be
   used. The size of the search space grows exponentially with this factor, and
   this size may become too large to perform exact nearest-neighbour searches.
#. The error that the Solovay-Kitaev algorithm can handle. The paper
   `The Solovay-Kitaev algorithm, Christopher M. Dawson, Michael A. Nielsen, \
   2005 <https://arxiv.org/abs/quant-ph/0505030>`_ computed a maximum allowable
   error of 1/32. This needs to be investigated.
#. The computational time gain: are the ANN queries faster than NN queries for
   our dataset (high dimensionality, reasonably high number of data).
"""

import copy
import os.path
import pickle
import typing

import annoy
import numpy
import scipy.spatial

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.utils.constants.others as qconsts
import qtoolkit.utils.types as qtypes


class NearestNeighbourStructure:
    """A generic and efficient data structure for nearest-neighbour requests."""

    def __init__(self, data_size: int) -> None:
        """Initialise the NearestNeighbourStructure instance.

        :param data_size: Length of item vector that will be indexed
        """
        self._annoy_index = annoy.AnnoyIndex(data_size, metric='euclidean')
        self._scipy_kdtree = None
        self._scipy_data = list()
        self._quantum_circuits = list()
        self._tree_number = -1

    def add_item(self, index: int,
                 quantum_circuit: qcirc.QuantumCircuit) -> None:
        """Add the given `quantum_circuit` to the indexed items.

        :param index: Unique index for the given quantum circuit. See the
            `Annoy documentation \
            <https://github.com/spotify/annoy#full-python-api>`_ for more
            information.
        :param quantum_circuit: The quantum circuit that needs to be added to
            the search space.
        """
        matrix = quantum_circuit.matrix
        vector = numpy.concatenate((numpy.real(matrix).reshape((-1, 1)),
                                    numpy.imag(matrix).reshape((-1, 1))))
        self._annoy_index.add_item(index, vector)
        self._scipy_data.append(vector)
        self._quantum_circuits.append(copy.copy(quantum_circuit).compress())

    def build(self, tree_number: int = 10) -> None:
        """Build the nearest-neighbour structure.

        This method should be called only once and only when all the item
        composing the search space have been added with
        :py:meth:`~.NearestNeighbourStructure.add_item`. Once this method has
        been called, items can no longer be added to the search space.

        :param tree_number: Parameter controlling the precision of the ANN and
            the computational cost of each ANN query. See the `Annoy \
            documentation <https://github.com/spotify/annoy#full-python-api>`_
            for more information.
        """
        self._tree_number = tree_number
        self._annoy_index.build(tree_number)
        self._scipy_data = numpy.array(self._scipy_data).reshape(
            (len(self._quantum_circuits), -1))
        self._scipy_kdtree = scipy.spatial.cKDTree(self._scipy_data)

    def save(self, filename: str) -> None:
        """Save the underlying NN structures on disk.

        This method will create 2 files:

        1. "`filename`": save of the Annoy data structure.
        2. "`filename`.circ": save of the compressed
           :py:class:`~.QuantumCircuit`.

        :param filename: The filename used to save the data.
        """
        filepath = os.path.join(qconsts.data_dir, filename)
        self._annoy_index.save(filepath)
        with open(filepath + '.circ', 'wb') as of:
            pickle.dump(self._quantum_circuits, of)

    def load(self, filename: str) -> None:
        """Reconstruct the underlying NN structures from a file on disk.

        :param filename: The filename used to save the data.
        """
        filepath = os.path.join(qconsts.data_dir, filename)
        self._annoy_index.load(filepath)
        with open(filepath + '.circ', 'rb') as input_file:
            self._quantum_circuits = pickle.load(input_file)

        n = len(self._quantum_circuits)
        dim = 2 ** self._quantum_circuits[0].qubit_number
        m = 2 * dim ** 2
        self._scipy_data = numpy.zeros((n, m))
        for idx, circuit in enumerate(self._quantum_circuits):
            matrix = circuit.matrix
            vector = numpy.concatenate((numpy.real(matrix).reshape((1, -1)),
                                        numpy.imag(matrix).reshape((1, -1))),
                                       axis=1)
            self._scipy_data[idx] = vector
        self._scipy_kdtree = scipy.spatial.cKDTree(self._scipy_data)

    def query(self, matrix: qtypes.UnitaryMatrix) -> typing.Tuple[
        float, qcirc.QuantumCircuit]:
        """Query the underlying data structure for nearest-neighbour of matrix.

        .. warning::
           For the moment this method perform 2 queries: one NN query and one
           ANN query. See :py:mod:`~.nearest_neighbour_structure` docstring for
           more information.

        :param matrix: The matrix we are searching an approximation for.
        :return: The distance of the found approximation along with the index of
            the approximation.
        """
        vector = numpy.concatenate((numpy.real(matrix).reshape((-1, 1)),
                                    numpy.imag(matrix).reshape((-1, 1))))
        # Nearest neighbours
        nns, dists = self._annoy_index.get_nns_by_vector(vector, 1,
                                                         include_distances=True)
        dists_scipy, nns_scipy = self._scipy_kdtree.query(vector.reshape((-1,)),
                                                          1)
        # # BRUTEFORCE
        # distances = numpy.linalg.norm(
        #     self._bruteforce_data - su2_trans.su2_to_so3(matrix), axis=1)
        # idx = numpy.argmin(distances)

        return dists_scipy, self._quantum_circuits[nns_scipy].uncompress()
