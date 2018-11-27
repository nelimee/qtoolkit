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

"""Implementation of the group_commutator for :math:`SU(d)` matrices."""

import typing

import numpy

import qtoolkit.maths.matrix.su2.group_commutator as gc_su2
import qtoolkit.utils.types as qtypes


def group_commutator(
    U: qtypes.SUdMatrix
) -> typing.Tuple[qtypes.SUdMatrix, qtypes.SUdMatrix]:
    """Finds :math:`V, W \\in U(d) \\mid U = V W V^\\dagger W^\\dagger`.

    .. note::
       The implementation of this method is based on `this implementation by
       Paul Pham \
       <https://github.com/cryptogoth/skc-python/blob/master/skc/group_factor.py#L75>`.

    :param U: The unitary matrix in :math:`U(d)` to decompose.
    :return: a tuple containing (:math:`V`, :math:`W`).
    """
    dim = U.shape[0]

    # We diagonalise the matrix.
    eigvals, eigvecs = numpy.linalg.eig(U)

    Vt = numpy.identity(dim, dtype=numpy.complex)
    Wt = numpy.identity(dim, dtype=numpy.complex)
    # We construct the 2*2 diagonal matrices from the eigenvalues.
    for i in range(dim // 2):
        U_i = numpy.diag(eigvals[2 * i : 2 * (i + 1)])
        V_i, W_i = gc_su2.group_commutator(U_i)
        a, b = 2 * i, 2 * i + 1
        Vt[a : b + 1, a : b + 1] = V_i
        Wt[a : b + 1, a : b + 1] = W_i
    V, W = eigvecs @ Vt @ eigvecs.T.conj(), eigvecs @ Wt @ eigvecs.T.conj()

    return V, W
