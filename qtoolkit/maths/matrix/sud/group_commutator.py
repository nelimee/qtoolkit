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

"""Implementation of the group_commutator decomposition for SU(d) matrices."""

import typing

import numpy
import scipy

import qtoolkit.maths.matrix.distances as qdists
import qtoolkit.maths.matrix.su2.group_commutator as gc_su2
import qtoolkit.utils.types as qtypes


def fourier_matrix(dim: int) -> qtypes.UnitaryMatrix:
    i, j = numpy.meshgrid(numpy.arange(dim), numpy.arange(dim))
    omega = numpy.exp(- 2 * numpy.pi * 1.j / dim)
    W = numpy.power(omega, i * j) / numpy.sqrt(dim)
    return W


def group_commutator(U: qtypes.SUdMatrix) -> typing.Tuple[
    qtypes.SUdMatrix, qtypes.SUdMatrix]:
    dim = U.shape[0]

    # We diagonalise the matrix.
    eigvals, eigvecs = numpy.linalg.eig(U)

    Vt = numpy.identity(dim, dtype=numpy.complex)
    Wt = numpy.identity(dim, dtype=numpy.complex)
    # We construct the 2*2 diagonal matrices from the eigenvalues.
    for i in range(dim // 2):
        U_i = numpy.diag(eigvals[2 * i:2 * (i + 1)])
        V_i, W_i = gc_su2.su2_group_commutator_decompose(U_i)
        Vt[2 * i, 2 * i], Vt[2 * i + 1, 2 * i + 1] = V_i[0, 0], V_i[1, 1]
        Wt[2 * i, 2 * i], Wt[2 * i + 1, 2 * i + 1] = W_i[0, 0], W_i[1, 1]

    V, W = eigvecs @ Vt @ eigvecs.T.conj(), eigvecs @ Wt @ eigvecs.T.conj()

    return V, W


def group_commutator_2(U: qtypes.SUdMatrix) -> typing.Tuple[
    qtypes.SUdMatrix, qtypes.SUdMatrix]:
    dim = U.shape[0]

    epsilon = qdists.operator_norm(numpy.identity(dim) - U)
    C1 = 4
    Cgc2 = numpy.sqrt(numpy.sqrt(dim) * (dim - 1) / 2)
    Cgc1 = C1 * Cgc2 ** 3

    # Compute the matrix H
    H = - 1.j * scipy.linalg.logm(U)
    # Find its eigenvalues
    eigs = scipy.linalg.eigvals(H)
    # Express H in the Fourier basis:
    W = fourier_matrix(dim)
    Hf = W @ numpy.diag(eigs) @ W.T.conj()

    # Compute F and G.
    G_diag = numpy.arange(dim) - (dim - 1) / 2
    G = numpy.diag(G_diag)
    G_diag = G_diag.reshape((-1, 1))
    F = 1.j * Hf / (G_diag - G_diag.T + numpy.diag(
        float('inf') * numpy.ones((G_diag.shape[0],))))

    # Rescale F and G
    norm_Hf = qdists.operator_norm(Hf)
    coeff = numpy.sqrt(numpy.sqrt(dim)) * numpy.sqrt(
        qdists.operator_norm(Hf)) / numpy.sqrt((dim - 1) / 2)
    F /= coeff
    G *= coeff

    V, W = scipy.linalg.expm(1.j * F), scipy.linalg.expm(1.j * G)

    return V, W
