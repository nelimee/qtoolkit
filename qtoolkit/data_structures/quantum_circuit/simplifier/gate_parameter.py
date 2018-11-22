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

"""Implementation of the :py:class:`~.GateParameter` class."""

import typing


class GateParameter:
    """Stores the relationship between multiple parameters in a
    :py:class:`~.SimplificationRule`.

    This class stores 2 attributes:

    1. A hashable identifier.
    2. A function taking a float as parameter and returning another float.

    """

    def __init__(self, rule_identifier: typing.Hashable,
                 transformation: typing.Callable[[float], float] = lambda
                     x: x) -> None:
        """Initialise a :py:class:`~.GateParameter` instance.

        :param rule_identifier: An identifier that will be shared by all
            instances of :py:class:`~.GateParameter` that should match each
            other.
        :param transformation:
        """
        self._id = rule_identifier
        self._transformation = transformation

    def apply_transformation(self, non_transformed_value: float) -> float:
        return self._transformation(non_transformed_value)

    @property
    def id(self):
        return self._id
