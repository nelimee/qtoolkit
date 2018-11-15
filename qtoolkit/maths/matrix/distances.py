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

"""Implements several distances."""

import numpy
import scipy

import qtoolkit.utils.types as qtypes


def fowler_distance(A: qtypes.UnitaryMatrix, B: qtypes.UnitaryMatrix) -> float:
    """Computes the Fowler distance between A and B.

    The Fowler distance is a distance that does not depend on the relative phase
    between the 2 matrices.

    :param A: First matrix.
    :param B: Second matrix.
    :return: The Fowler distance between A and B.
    """
    dimension = A.shape[0]
    product = A.T.conj() @ B
    trace = numpy.trace(product)
    return numpy.sqrt(numpy.abs(dimension - trace) / dimension)


def fowler_distances(A: numpy.ndarray,
                     B: qtypes.UnitaryMatrix) -> numpy.ndarray:
    """Computes the Fowler distances between each A[i] and B.

    This method is an optimised version of
        N = A.shape[0]
        distances = []
        for i in range(N):
            distances.append(fowler_distance(A[i], B))
        return numpy.array(distances)

    :param A: a 3-dimensional array of dimensions (N, m, m) with N the number
    of unitary matrices of size (m, m) to process.
    :param B: a unitary matrix of size (m, m).
    :return: a 1-dimensional array of size N containing the fowler distances
    between A[i] and B for 0 <= i < N.
    """
    dimension = B.shape[0]
    products = numpy.transpose(A, axes=(0, 2, 1)).conj() @ B
    traces = numpy.trace(products, axis1=1, axis2=2)
    frac = dimension - numpy.abs(traces) / dimension
    return numpy.sqrt(numpy.abs(frac))


def trace_distance(A: qtypes.UnitaryMatrix, B: qtypes.UnitaryMatrix) -> float:
    """Computes the trace distance between A and B.

    :param A: First matrix.
    :param B: Second matrix.
    :return: The trace distance between A and B.
    """
    difference = A - B
    eigs = numpy.linalg.eigvals(difference @ difference.T.conj())
    return numpy.linalg.norm(eigs)


def operator_norm(U: qtypes.UnitaryMatrix) -> float:
    """Operator norm of a unitary matrix U.

    ||U||_{op} = sup {||Uv|| : ||v||=1}

    :param U: A unitary matrix.
    :return: The operator norm of U.
    """
    eigenvalues = scipy.linalg.eigvals(U)
    return numpy.max(numpy.abs(eigenvalues))
