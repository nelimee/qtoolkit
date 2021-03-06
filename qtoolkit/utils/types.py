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

"""Defines type hints for the qtoolkit project."""

import typing

import numpy
import qiskit
import scipy.sparse
import sympy

# Qiskit-related types
QuantumGateParameter = typing.Union[float, int, complex, sympy.Basic]
GateParameterTransformation = typing.Callable[
    [QuantumGateParameter], QuantumGateParameter
]

QuantumBit = typing.Tuple[qiskit.QuantumRegister, int]
ClassicalBit = typing.Tuple[qiskit.ClassicalRegister, int]
QuantumGateArgument = typing.Union[
    qiskit.QuantumRegister, qiskit.ClassicalRegister, QuantumBit, ClassicalBit
]
# TODO: change this type name.
QuantumInstructions = typing.Union[qiskit.QuantumCircuit, qiskit.CompositeGate]

# Mathematical types
GenericMatrix = typing.Union[numpy.ndarray, scipy.sparse.spmatrix]
UnitaryMatrix = GenericMatrix
HermitianMatrix = GenericMatrix
SU2Matrix = GenericMatrix

SO3Vector = numpy.ndarray
SUdMatrix = GenericMatrix

SUdMatrixGenerator = typing.Union[SUdMatrix, typing.Callable[..., SUdMatrix]]
UnitaryMatrixGenerator = typing.Callable[[numpy.ndarray], UnitaryMatrix]
SUMatrix = GenericMatrix

GenericArray = typing.Union[numpy.ndarray, scipy.sparse.spmatrix]

UnsignedIntegerType = typing.Union[
    typing.Type[numpy.uint8],
    typing.Type[numpy.uint16],
    typing.Type[numpy.uint32],
    typing.Type[numpy.uint64],
]

# Other types
NearestNeighbourQueryable = typing.Any

Bounds = typing.Optional[typing.Sequence[typing.Optional[numpy.ndarray]]]
