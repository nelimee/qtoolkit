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

import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq
import qtoolkit.utils.types as qtypes


class NearestNeighbourStructure:
    """A generic and efficient data structure for nearest-neighbour requests."""

    def __init__(self, matrices: numpy.ndarray, gate_sequences: numpy.ndarray,
                 basis: typing.Sequence[qtypes.SUdMatrix]) -> None:
        """Initialise the NNStructure instance.

        :param matrices: An array with shape (n,m) storing the n data points of
        dimension m to be indexed. This array is **not** copied, modifications
        will result in bogus results.
        """
        self._matrices = matrices
        self._gate_sequences = gate_sequences
        self._basis = basis

    def query(self, x: numpy.ndarray) -> typing.Tuple[
        float, qgate_seq.QuantumGateSequence]:
        """Query the underlying data structure for nearest-neighbour of X.

        For the moment, this method performs a brute-force search.

        :param x: The point we are searching an approximation for.
        :return: The distance of the found approximation along with the
        index of the approximation.
        """
        # distances = qdists.fowler_distances(self._matrices, x)
        distances = numpy.linalg.norm(self._matrices - x, axis=(1, 2))
        index = numpy.argmin(distances)
        dist = distances[index]

        return dist, qgate_seq.QuantumGateSequence(self._basis,
                                                   self._gate_sequences[index])
