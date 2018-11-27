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

"""Frequent matrices in quantum computing.

This file contains several matrices that are frequently used in quantum
computing.
"""

import numpy

import qtoolkit.utils.types as qtypes

######################################
#           PAULI MATRICES           #
######################################
# Definitions with sigma
SIGMA_X = numpy.array([[0, 1], [1, 0]], dtype=numpy.complex)
SIGMA_Y = numpy.array([[0, -1.j], [1.j, 0]], dtype=numpy.complex)
SIGMA_Z = numpy.array([[1, 0], [0, -1]], dtype=numpy.complex)
SIGMA_X_SU2 = SIGMA_X / numpy.lib.scimath.sqrt(numpy.linalg.det(SIGMA_X))
SIGMA_Y_SU2 = SIGMA_Y / numpy.lib.scimath.sqrt(numpy.linalg.det(SIGMA_Y))
SIGMA_Z_SU2 = SIGMA_Z / numpy.lib.scimath.sqrt(numpy.linalg.det(SIGMA_Z))
# Aliases to the definitions with sigma
P_X = SIGMA_X
P_Y = SIGMA_Y
P_Z = SIGMA_Z
P_X_SU2 = SIGMA_X_SU2
P_Y_SU2 = SIGMA_Y_SU2
P_Z_SU2 = SIGMA_Z_SU2


######################################
#           IDENTITY MATRIX          #
######################################
IDENTITY_2X2 = numpy.identity(2, dtype=numpy.complex)
ID2 = IDENTITY_2X2
# In SU(2)
IDENTITY_2X2_SU2 = IDENTITY_2X2
ID2_SU2 = ID2


######################################
#       QUANTUM GATES MATRICES       #
######################################
X = numpy.array([[0, 1], [1, 0]], dtype=numpy.complex)
Y = numpy.array([[0, -1.j], [1.j, 0]], dtype=numpy.complex)
Z = numpy.array([[1, 0], [0, -1]], dtype=numpy.complex)
H = numpy.array([[1, 1], [1, -1]], dtype=numpy.complex) / numpy.sqrt(2)
S = numpy.array([[1, 0], [0, 1.j]], dtype=numpy.complex)
T = numpy.array([[1, 0], [0, numpy.exp(1.j * numpy.pi / 4)]],
                dtype=numpy.complex)
CX = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                 dtype=numpy.complex)


def Rx(parameters: numpy.ndarray) -> qtypes.SU2Matrix:
    theta = parameters[0]
    cos_theta_2 = numpy.cos(theta / 2)
    isin_theta_2 = -1.j * numpy.sin(theta / 2)
    return numpy.array(
        [[cos_theta_2, isin_theta_2], [isin_theta_2, cos_theta_2]])


def Ry(parameters: numpy.ndarray) -> qtypes.SU2Matrix:
    theta = parameters[0]
    cos_theta_2 = numpy.cos(theta / 2)
    sin_theta_2 = numpy.sin(theta / 2)
    return numpy.array(
        [[cos_theta_2, -sin_theta_2], [sin_theta_2, cos_theta_2]])


def Rz(parameters: numpy.ndarray) -> qtypes.SU2Matrix:
    theta = parameters[0]
    e_itheta_2 = numpy.exp(1.j * theta / 2)
    return numpy.array([[1 / e_itheta_2, 0], [0, e_itheta_2]])


# Quantum gates in SU(2)
X_SU2 = X / numpy.lib.scimath.sqrt(numpy.linalg.det(X))
Y_SU2 = Y / numpy.lib.scimath.sqrt(numpy.linalg.det(Y))
Z_SU2 = Z / numpy.lib.scimath.sqrt(numpy.linalg.det(Z))
H_SU2 = H / numpy.lib.scimath.sqrt(numpy.linalg.det(H))
S_SU2 = S / numpy.lib.scimath.sqrt(numpy.linalg.det(S))
T_SU2 = T / numpy.lib.scimath.sqrt(numpy.linalg.det(T))
CX_SU2 = CX / numpy.lib.scimath.sqrt(
    numpy.lib.scimath.sqrt(numpy.linalg.det(CX)))

P0 = numpy.array([[1, 0], [0, 0]], dtype=numpy.complex)
P1 = numpy.array([[0, 0], [0, 1]], dtype=numpy.complex)
