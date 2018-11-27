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

import numpy

import qtoolkit.maths.matrix.su2.transformations as su2trans
import qtoolkit.utils.types as qtypes


def similarity_matrix(A: qtypes.SU2Matrix, B: qtypes.SU2Matrix) -> qtypes.SU2Matrix:
    """Find :math:`S \\in SU(2) \\mid A = S B S^\\dagger`.

    :param A: First :math:`SU(2)` matrix.
    :param B: Second :math:`SU(2)` matrix.
    :return: the :math:`SU(2)` matrix :math:`S`.
    """
    a, b = su2trans.su2_to_so3(A), su2trans.su2_to_so3(B)
    norm_a, norm_b = numpy.linalg.norm(a, 2), numpy.linalg.norm(b, 2)

    s = numpy.cross(b, a)
    norm_s = numpy.linalg.norm(s, 2)

    if norm_s == 0:
        # The representative vectors are too close to each other, this
        # means that the original matrices are also very close, and so
        # returning the identity matrix is fine.
        return numpy.identity(2)

    angle_between_a_and_b = numpy.arccos(numpy.inner(a, b) / (norm_a * norm_b))
    s *= angle_between_a_and_b / norm_s

    S = su2trans.so3_to_su2(s)
    return S
