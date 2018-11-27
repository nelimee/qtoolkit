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

"""Implementation of `the group-commutator algorithm of Christopher Dawson \
<https://github.com/cmdawson/sk/blob/master/src/su2.cpp#L25>`_."""

import typing

import numpy

import qtoolkit.maths.matrix.su2.similarity_matrix as sim_matrix
import qtoolkit.maths.matrix.su2.transformations as su2trans
import qtoolkit.utils.types as qtypes


def group_commutator(
    U: qtypes.UnitaryMatrix
) -> typing.Tuple[qtypes.SU2Matrix, qtypes.SU2Matrix]:
    """Finds :math:`V, W \\in U(2) \\mid U = V W V^\\dagger W^\\dagger`.

    :param U: The unitary matrix in :math:`U(2)` to decompose.
    :return: a tuple containing (:math:`V`, :math:`W`).
    """
    # unitary is a rotation of a unknown angle $\theta$ about some unknown
    # axis. Here, we find the angle $\theta$.
    so3_unitary = su2trans.su2_to_so3(U)
    theta = numpy.linalg.norm(so3_unitary, 2)

    # Then, we construct the matrix that consist of a rotation of $\theta$
    # about the X-axis.
    X_unitary = su2trans.so3_to_su2(numpy.array([theta, 0.0, 0.0]))
    # We find the similarity matrix between the original unitary and the
    # rotation about the X-axis we just created.
    S = sim_matrix.similarity_matrix(U, X_unitary)

    # Now we perform the real computations to find V and W, but we perform
    # them on the unitary rotating about the X-axis and not on the
    # original unitary.
    A, B = _X_axis_su2_group_commutator_decompose(X_unitary)

    # Compute the real V and W from A and B.
    V, W = S @ A @ S.T.conj(), S @ B @ S.T.conj()

    return V, W


def _X_axis_su2_group_commutator_decompose(
    Ux: qtypes.SU2Matrix
) -> typing.Tuple[qtypes.SU2Matrix, qtypes.SU2Matrix]:
    """Finds :math:`A, B \\in U(d) \\mid Ux = A B A^\\dagger B^\\dagger`.

    This method is restricted to matrices Ux that are rotations around the
    X-axis.
    From the analysis performed in http://arXiv.org/abs/quant-ph/0505030v2
    section 4.1, A and B can be seen as rotations.

    :param Ux: The unitary matrix in :math:`U(d)` to decompose.
    :return: a tuple containing (:math:`A`, :math:`B`).
    """
    # In the following code, theta is the angle of the given Ux, phi is the
    # angle of A and B.

    # We transform the input matrix as a vector of 4 real numbers because
    # these numbers are directly related to the cosinus and the sinus of
    # phi.
    unitary_cart4_coefficients = su2trans.su2_to_H(Ux)
    # From these coefficients, we have directly the value of cos(phi/2).
    cos_theta_2 = unitary_cart4_coefficients[0]

    # We know from http://arXiv.org/abs/quant-ph/0505030v2, Equation (10)
    # that theta and phi are linked.
    # Equation (10) can be reformulated as
    #    4*X^2 - 4*X + sin^2(theta/2) = 0
    # where X = sin^4(phi/2).
    # Solving this equation (which is easy because polynomial of degree 2)
    # gives X = (1 +/- cos(theta/2)) / 2 which can be rewritten as
    # sin^4(phi/2) = (1 +/- cos(theta/2)) / 2
    # The '+' version gives wrong results, but I don't know why. This should
    # be investigated when possible.
    sin_phi_2 = numpy.sqrt(numpy.sqrt((1 - cos_theta_2) / 2))
    cos_phi_2 = numpy.sqrt(1 - sin_phi_2 * sin_phi_2)

    # Compute the spherical coordinates of the vector representing the 3D
    # rotation.
    phi = 2 * numpy.arcsin(sin_phi_2)  # theta in the spherical system
    alpha = numpy.arctan(sin_phi_2)  # phi in the spherical system

    # Create the vector in cartesian coordinates representing the rotation
    # in 3D.
    # [spherical coordinate system] = [variables in this function].
    #                             r = phi
    #                         theta = phi/2
    #                           phi = alpha
    # We construct a and w such that the corresponding SU(2) matrices A
    # and W satisfy unitary_x == A @ W.
    w = numpy.array(
        [
            phi * sin_phi_2 * numpy.cos(alpha),
            phi * sin_phi_2 * numpy.sin(alpha),
            phi * cos_phi_2,
        ]
    )
    a = w.copy()
    a[2] = -w[2]

    # Construct the matrices A and W from a and w.
    A, W = su2trans.so3_to_su2(a), su2trans.so3_to_su2(w)

    # Finds B such that W = B @ A.T.conj() @ B.T.conj()
    # With such a B, A @ W == unitary_x == A @ B @ A.T.conj() @ B.T.conj()
    # which is exactly what we are searching for!
    B = sim_matrix.similarity_matrix(W, A.T.conj())

    # Return the matrices we were searching for.
    return A, B
