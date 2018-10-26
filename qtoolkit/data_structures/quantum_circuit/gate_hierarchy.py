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

"""Implement the gate hierarchy used by the QuantumCircuit class."""

import qtoolkit.utils.types as qtypes


class QuantumInstruction:
    """The most generic "gate" of the hierarchy."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self):
        return self._name


class QuantumGate(QuantumInstruction):
    """Base class for all the quantum gates."""

    def __init__(self, name: str, matrix: qtypes.SUdMatrix,
                 *parameters: float) -> None:
        super().__init__(name)
        self._matrix = matrix
        self._parameters = parameters

    @property
    def matrix(self):
        return self._matrix

    @property
    def dim(self):
        return self._matrix.shape[0]

    @property
    def parameters(self):
        return self._parameters


class ParametrisedQuantumGate(QuantumInstruction):

    def __init__(self, name: str,
                 matrix_generator: qtypes.SUdMatrixGenerator) -> None:
        super().__init__(name)
        self._matrix_generator = matrix_generator

    def __call__(self, *args, **kwargs) -> QuantumGate:
        return QuantumGate(self.name, self._matrix_generator(*args, **kwargs),
                           *args)
