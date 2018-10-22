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

"""A set of functions to generate SU(d) matrices."""

import numpy
import numpy.random as rand

import qtoolkit.utils.types as qtypes


def generate_random_SUd(dim: int) -> qtypes.SUdMatrix:
    """Generate a random matrix in SU(d).

    The algorithm implemented is presented in the article
    https://arxiv.org/pdf/math-ph/0609050.pdf.

    :param dim: dimension of the matrix to be generated.
    :return: A random SU(d) matrix distributed with Haar measure.
    """
    Z = (rand.rand(dim, dim) + 1.j * rand.rand(dim, dim)) / numpy.sqrt(2)
    Q, R = numpy.linalg.qr(Z)
    diag = numpy.diag(R)
    D = numpy.diag(diag / numpy.abs(diag))
    return Q @ D
