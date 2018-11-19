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

import typing

import numpy.random

import qtoolkit.utils.types as qtypes
from qtoolkit.data_structures.quantum_gate_sequence import QuantumGateSequence


def generate_random_gate_sequence(
    basis: typing.Sequence[qtypes.SUdMatrixGenerator], length: int,
    parameters_bounds: typing.Optional[
        numpy.ndarray] = None) -> QuantumGateSequence:
    """Generate a random gate sequence.

    :param basis: basis used to generate the sequence.
    :param length: length of the random sequence generated.
    :param parameters_bounds: bounds for the parameters. All the irrelevant
    parameters can be set to any value and will not be accessed.
    :return: a random quantum gate sequence.
    """
    sequence = numpy.random.randint(0, len(basis), size=length)
    if parameters_bounds is not None:
        a, b = parameters_bounds[0], parameters_bounds[1]
        parameters = numpy.random.rand(length) * (b[sequence] - a[sequence]) + \
                     a[sequence]

    else:
        parameters = None

    return QuantumGateSequence(basis, sequence, parameters)
