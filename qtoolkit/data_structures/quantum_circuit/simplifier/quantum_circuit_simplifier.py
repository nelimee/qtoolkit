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

"""Implement classes related to :py:class:`~.QuantumCircuit` simplification.

An instance of :py:class:`~.QuantumCircuit` is in essence a sequence of
quantum gates applied in order. Some sub-sequences may acts on the quantum state
as the identity matrix or as a matrix that is implementable with a shorter
sequence of quantum gates. These sub-sequences are called "simplifiable
sequences".

For the moment, this module only implements "detection" and not "simplification"
of simplifiable sequences.
"""

import typing

from qtoolkit.data_structures.quantum_circuit.quantum_circuit import \
    QuantumCircuit
from qtoolkit.data_structures.quantum_circuit.simplifier.simplification_rule \
    import \
    SimplificationRule


class QuantumCircuitSimplificationDetector:
    """Simplifiable sequence detector for :py:class:`~.QuantumCircuit`
    instances.

    This class only detects simplifiable sequences. All the detection part is
    done in the rules, this class only collects rules and call the appropriate
    method for each stored rule to determine if the given instance of
    :py:class:`~.QuantumCircuit` is simplifiable according to at least one rule.
    """

    def __init__(self,
                 simplifications: typing.List[SimplificationRule]) -> None:
        """Simplifiable sequence detector for :py:class:`~.QuantumCircuit`
        instances.

        This class only detects simplifiable sequences. All the detection part
        is done in the rules, this class only collects rules and call the
        appropriate method for each stored rule to determine if the given
        instance of :py:class:`~.QuantumCircuit` is simplifiable according to at
        least one rule.

        :param simplifications: A list of :py:class:`~.SimplificationRule` that
            will be added at initialisation. You can equivalently call
            :py:meth:`~.QuantumCircuitSimplificationDetector.add_rule` for each
            rule in `simplifications`.
        """
        self._rules = list()
        for rule in simplifications:
            self.add_rule(rule)

    def add_rule(self, rule: SimplificationRule) -> None:
        """Add a rule to the list of stored rules.

        :param rule: The rule to add.
        """
        self._rules.append(rule)

    def is_simplifiable(self, qcirc: QuantumCircuit) -> bool:
        """Check if `qcirc` is simplifiable according to the stored rules.

        :param qcirc: The quantum circuit to check for simplifiability.
        :return: True if `qcirc` is simplifiable according to the rules stored
            by this instance.
        """
        for rule in self._rules:
            if rule.is_simplifiable(qcirc):
                return True
        return False

    def is_simplifiable_from_last(self, qcirc: QuantumCircuit) -> bool:
        """Check if the last inserted operation of `qcirc` made it simplifiable.

        This method only check if the last inserted operation in `qcirc`
        introduced a simplifiable sequence.

        :param qcirc: The quantum circuit to check for simplifiability.
        :return: True if the last inserted operation in `qcirc` introduced a
            simplifiable sequence according to the rules stored by this
            instance.
        """
        for rule in self._rules:
            if rule.is_simplifiable_from_last(qcirc):
                return True
        return False
