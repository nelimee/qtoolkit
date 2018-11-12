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

import copy
import typing

import annoy
import numpy

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.utils.types as qtypes


class NearestNeighbourStructure:
    """A generic and efficient data structure for nearest-neighbour requests."""

    def __init__(self, data_size: int) -> None:
        """Initialise the NearestNeighbourStructure instance.

        :param data_size: Length of item vector that will be indexed
        """
        self._annoy_index = annoy.AnnoyIndex(data_size)
        self._quantum_circuits = list()

    def add_item(self, index: int,
                 quantum_circuit: qcirc.QuantumCircuit) -> None:
        matrix = quantum_circuit.matrix
        vector = numpy.concatenate((numpy.real(matrix).reshape((-1, 1)),
                                    numpy.imag(matrix).reshape((-1, 1))))
        self._annoy_index.add_item(index, vector)
        self._quantum_circuits.append(copy.copy(quantum_circuit).compress())

    def build(self, tree_number: int = 10) -> None:
        self._annoy_index.build(tree_number)

    def save(self, filename: str) -> None:
        self._annoy_index.save(filename)

    def load(self, filename: str) -> None:
        self._annoy_index.load(filename)

    def query(self, matrix: qtypes.UnitaryMatrix) -> typing.Tuple[
        float, qcirc.QuantumCircuit]:
        """Query the underlying data structure for nearest-neighbour of matrix.

        :param matrix: The matrix we are searching an approximation for.
        :return: The distance of the found approximation along with the index of
        the approximation.
        """
        vector = numpy.concatenate((numpy.real(matrix).reshape((-1, 1)),
                                    numpy.imag(matrix).reshape((-1, 1))))
        nns, dists = self._annoy_index.get_nns_by_vector(vector, 1,
                                                         include_distances=True)

        return dists[0], self._quantum_circuits[nns[0]]
