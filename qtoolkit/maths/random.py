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

"""Routines related to random generation."""

import numpy


def generate_random_complex(amplitude: float = 1.0) -> complex:
    """Generate a random complex number.

    :param amplitude: The amplitude of the complex number to generate.
    :return: a random complex number of the desired amplitude.
    """
    coefficients = numpy.random.rand(2)
    norm = numpy.linalg.norm(coefficients)
    return (coefficients[0] + 1.0j * coefficients[1]) * amplitude / norm


def generate_random_normalised_complexes(size: int) -> numpy.ndarray:
    """Generate size complex numbers as a normalised vector.

    :param size: A positive integer.
    :return: a normalised list of size random complex numbers.
    """
    complexes = numpy.random.rand(size) + 1.0j * numpy.random.rand(size)
    return complexes / numpy.linalg.norm(complexes)


def generate_unit_norm_quaternion() -> numpy.ndarray:
    """Generate a random normalised quaternion.

    :return: A random normalised quaternion.
    """
    quaternion = numpy.random.rand(4)
    return quaternion / numpy.linalg.norm(quaternion)
