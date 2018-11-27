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

import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.utils.types as qtypes


def fowler_distance(A: qtypes.UnitaryMatrix, B: qtypes.UnitaryMatrix) -> float:
    """Computes the Fowler distance between :math:`A` and :math:`B`.

    The Fowler distance is implemented in `Paul Pham's skc-python repository \
    <https://github.com/cryptogoth/skc-python/blob/master/skc/group_factor.py#L75>`_.

    :param A: First matrix.
    :param B: Second matrix.
    :return: the Fowler distance between :math:`A` and :math:`B`.
    """
    dimension = A.shape[0]
    product = A.T.conj() @ B
    trace = numpy.trace(product)
    return numpy.sqrt(numpy.abs(dimension - trace) / dimension)


def fowler_distances(A: numpy.ndarray, B: qtypes.UnitaryMatrix) -> numpy.ndarray:
    """Computes the Fowler distances between each :math:`A[i]` and :math:`B`.

    This method is an optimised version of

    .. highlight:: python
    .. code-block:: python

        N = A.shape[0]
        distances = []
        for i in range(N):
            distances.append(fowler_distance(A[i], B))
        return numpy.array(distances)

    :param A: a 3-dimensional array of dimensions (:math:`N`, :math:`m`,
        :math:`m`) with :math:`N` the number of unitary matrices of size
        (:math:`m`, :math:`m`) to process.
    :param B: a unitary matrix of size (:math:`m`, :math:`m`).
    :return: a 1-dimensional array of size :math:`N` containing the fowler
        distances between :math:`A[i]` and :math:`B` for
        :math:`0 \\leqslant i < N`.
    """
    dimension = B.shape[0]
    products = numpy.transpose(A, axes=(0, 2, 1)).conj() @ B
    traces = numpy.trace(products, axis1=1, axis2=2)
    frac = dimension - numpy.abs(traces) / dimension
    return numpy.sqrt(numpy.abs(frac))


def trace_distance(A: qtypes.UnitaryMatrix, B: qtypes.UnitaryMatrix) -> float:
    """Computes the trace distance between :math:`A` and :math:`B`.

    :param A: First matrix.
    :param B: Second matrix.
    :return: the trace distance between :math:`A` and :math:`B`.
    """
    difference = A - B
    eigs = numpy.linalg.eigvals(difference @ difference.T.conj())
    return numpy.linalg.norm(eigs)


def operator_norm(U: qtypes.UnitaryMatrix) -> float:
    """Operator norm of a unitary matrix :math:`U`.

    .. math::

        ||U||_{op} = \\sup_{||v||=1} \\left\\{||Uv||\\right\\}

    :param U: A unitary matrix.
    :return: the operator norm of :math:`U`.
    """
    eigenvalues = scipy.linalg.eigvals(U)
    return numpy.max(numpy.abs(eigenvalues))


def gloa_objective_function(
    gate_sequence: qcirc.QuantumCircuit,
    U: qtypes.UnitaryMatrix,
    correctness_weight: float,
    circuit_cost_weight: float,
    circuit_cost_func: qcirc.CircuitCostFunction,
) -> float:
    """Compute a modified GLOA objective function.

    The GLOA article is: https://arxiv.org/abs/1004.2242.

    This objective function has been modified because the one presented in the
    GLOA article may make the algorithm converge to a matrix :math:`M` such that
    :math:`U  M^\\dagger = - I_n`. This new objective function prevent this
    problem.

    :param gate_sequence: the sequence of quantum gate that is candidate to
        approximate the objective_unitary matrix.
    :param U: the unitary matrix we are searching an approximation for.
    :param correctness_weight: importance of the correctness of the circuit
        in the objective function. Corresponds to the parameter alpha in the
        original paper.
    :param circuit_cost_weight: importance of the circuit cost in the
        objective function. Corresponds to the parameter beta in the original
        paper.
    :param circuit_cost_func: a function that will associate a cost to a given
        sequence of quantum gates.
    :return: the fidelity of the approximation.
    """
    N = U.shape[0]
    UUT = U @ gate_sequence.matrix.T.conj()
    trace_fidelity = (1 + numpy.trace(UUT) / N) / 2
    correctness = correctness_weight * trace_fidelity
    circuit_cost = circuit_cost_weight / circuit_cost_func(gate_sequence)
    return numpy.abs(1 - (correctness + circuit_cost))
