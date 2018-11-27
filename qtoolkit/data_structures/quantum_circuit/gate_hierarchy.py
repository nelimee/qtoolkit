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

"""Implementation of the gate hierarchy used by the QuantumCircuit class.

This module implements classes used to represent 1-qubit gates (parametrised or
not).

.. note::
    For n-qubit (:math:`n > 1`) gates see :py:class:`~.QuantumOperation`.
"""

import typing

import numpy

import qtoolkit.utils.types as qtypes


class QuantumInstruction:
    """The base class for all the gate hierarchy."""

    def __init__(self, name: str) -> None:
        """Initialise the QuantumInstruction instance.

        :param name: the name of the instruction.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Getter for the instruction name."""
        return self._name


class ParametrisedQuantumGate(QuantumInstruction):
    """Class representing a parametrised quantum gate."""

    def __init__(
        self,
        name: str,
        matrix_generator: qtypes.UnitaryMatrixGenerator,
        inverse: typing.Callable[["QuantumGate"], "QuantumGate"],
    ) -> None:
        """Initialise the parametrised quantum gate.

        :param name: the name of the quantum gate.
        :param matrix_generator: a callable that will generate a unitary matrix
            from a numpy.ndarray of floating-point value(s).
        :param inverse: a callable that will be forwarded to the
            :py:class:`~.QuantumGate` constructed from this
            :py:class:`~.ParametrisedQuantumGate`. See
            :py:meth:`.QuantumGate.inverse` for more information.
        """
        super().__init__(name)
        self._matrix_generator = matrix_generator
        self._inverse_callable = inverse

    def __call__(self, *args: float, **kwargs) -> "QuantumGate":
        """Generate the quantum gate obtained with the given parameters.

        :param args: the parameters forwarded to the `matrix_generator`
            callable.
        :param kwargs: additional data forwarded to the `matrix_generator`
            callable.
        :return: a :py:class:`~.QuantumGate` corresponding to the current
        parametrised quantum gate with the given parameters.
        """
        params = numpy.array(args)
        return QuantumGate(
            self.name,
            self._matrix_generator(params, **kwargs),
            self._inverse_callable,
            parameters=params,
        )


class QuantumGate(QuantumInstruction):
    """Class representing a quantum gate."""

    def __init__(
        self,
        name: str,
        matrix: qtypes.UnitaryMatrix,
        inverse: typing.Callable[["QuantumGate"], "QuantumGate"],
        parameters: typing.Optional[numpy.ndarray] = None,
    ) -> None:
        """Initialise the :py:class:`~.QuantumGate` instance.

        :param name: name of the quantum gate.
        :param matrix: unitary matrix representing the quantum gate.
        :param inverse: callable used to inverse the current quantum gate.
        :param parameters: the set of parameters used to generate the current
            quantum gate. If the current quantum gate was not generated from a
            :py:class:`~.ParametrisedQuantumGate`, this value is empty.
        """
        super().__init__(name)
        self._matrix = matrix
        self._parameters = parameters
        self._inverse_callable = inverse

    @property
    def matrix(self):
        """Getter on the underlying unitary matrix."""
        return self._matrix

    @property
    def dim(self):
        """Getter on the dimension of the quantum gate."""
        return self._matrix.shape[0]

    @property
    def parameters(self):
        """Getter on the stored parameters."""
        return self._parameters

    @property
    def H(self):
        """Hermitian conjugate operator."""
        return self._inverse_callable(self)
