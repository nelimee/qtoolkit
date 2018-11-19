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

"""Transformations related to SU(2) matrices."""

import numpy

import qtoolkit.utils.constants.matrices as mconsts
import qtoolkit.utils.types as qtypes


def su2_to_so3(unitary: qtypes.SU2Matrix) -> qtypes.SO3Vector:
    r"""Convert a SU(2) matrix into x in e^{-ix.sigma/2}.

    This function is based on the formula:
    $$e^{ia (\hat{n} \dot \vec{\sigma})} =
    \cos(a) I + i(\hat{n} \dot \vec{\sigma}) \sin(a)$$
    from https://en.wikipedia.org/wiki/Pauli_matrices

    :param unitary: The SU(2) matrix to convert.
    :return: 3 real numbers parametrising entirely the given matrix.
    """
    # The coefficients are: $[-\sin(a) n_1, -\sin(a) n_2, -\sin(a) n_3]$.
    # The minus appears because there is a minus in the exponential we
    # convert to, which is not present in the formula given in the link.
    coefficients = numpy.array(
        [-numpy.imag(unitary[0, 1]), numpy.real(unitary[1, 0]),
         numpy.imag((unitary[1, 1] - unitary[0, 0]) / 2.0)])

    cos_theta_2 = numpy.real((unitary[0, 0] + unitary[1, 1]) / 2.0)
    # The vector $\hat{n}$ is supposed to be of unit-length so its norm
    # is 1. That is why the expression below gives the sinus.
    sin_theta_2 = numpy.linalg.norm(coefficients, 2)

    if sin_theta_2 == 0.0:
        coefficients = numpy.array([2 * numpy.arccos(cos_theta_2), 0.0, 0.0])
    else:
        # We return the vector of coefficients, not normalised.
        theta = 2 * numpy.arctan2(sin_theta_2, cos_theta_2)
        coefficients = theta * coefficients / sin_theta_2

    return coefficients


def so3_to_su2(coefficients: qtypes.SO3Vector) -> qtypes.SU2Matrix:
    r"""Convert a set of 3 real coefficients to a unique SU(2) matrix.

    Computes the unitary matrix in SU(2) with the formula
    $$e^{ia (\hat{n} \dot \vec{\sigma})} =
    \cos(a) I + i(\hat{n} \dot \vec{\sigma}) \sin(a)$$
    from https://en.wikipedia.org/wiki/Pauli_matrices

    :param coefficients: 3 real numbers characterising the SU(2) matrix.
    :return: The SU(2) matrix characterised by the given coefficients.
    """
    theta = numpy.linalg.norm(coefficients, 2)
    identity = numpy.identity(2)

    if theta == 0.0:
        return identity
    else:
        normalised_coefficients = coefficients / theta
        theta_2 = theta / 2
        sin_theta_2 = numpy.sin(theta_2)
        unitary = (numpy.cos(theta_2) * identity - 1.j * sin_theta_2 * (
            normalised_coefficients[0] * mconsts.P_X + normalised_coefficients[
            1] * mconsts.P_Y + normalised_coefficients[2] * mconsts.P_Z))
        return unitary


def su2_to_H(unitary: qtypes.SU2Matrix) -> numpy.ndarray:
    """Convert a SU(2) matrix to a unit-norm quaternion.

    Quaternions are composed of 4 real numbers. These numbers corresponds to
    the coefficients $\alpha_i$ in the equation
    $$e^{ia (\hat{n} \dot \vec{\sigma})} =
    \alpha_0 I + i(\hat{\alpha} \dot \vec{\sigma})$$
    from https://en.wikipedia.org/wiki/Pauli_matrices

    :param unitary: The unitary matrix to decompose.
    :return: The 4 real coefficients of the unit-norm quaternion representing
    the matrix.
    """
    coefficients = numpy.array(
        [numpy.real(unitary[0, 0]), -numpy.imag(unitary[0, 1]),
         numpy.real(unitary[1, 0]), numpy.imag(unitary[1, 1])])
    return coefficients


def H_to_su2(coefficients: numpy.ndarray) -> qtypes.SU2Matrix:
    """Convert a unit-norm quaternion to a SU(2) matrix.

    Quaternions are composed of 4 real numbers. These numbers corresponds to
    the coefficients $\alpha_i$ in the equation
    $$e^{ia (\hat{n} \dot \vec{\sigma})} =
    \alpha_0 I + i(\hat{\alpha} \dot \vec{\sigma})$$
    from https://en.wikipedia.org/wiki/Pauli_matrices

    :param coefficients: The coefficients characterising the SU(2) matrix.
    :return: The SU(2) matrix corresponding to the given coefficients.
    """
    return coefficients[0] * mconsts.ID2 - 1.j * (
        coefficients[1] * mconsts.P_X + coefficients[2] * mconsts.P_Y +
        coefficients[3] * mconsts.P_Z)


def unitary_to_su2(unitary: qtypes.UnitaryMatrix) -> qtypes.SU2Matrix:
    """Project the given unitary matrix in SU(2).

    This routine just scales the given unitary to make its determinant equals to
    1.

    :param unitary: The unitary matrix to project.
    :return: The corresponding SU(2) matrix.
    """
    det = unitary[0, 0] * unitary[1, 1] - unitary[0, 1] * unitary[1, 0]
    return unitary / numpy.lib.scimath.sqrt(det)
