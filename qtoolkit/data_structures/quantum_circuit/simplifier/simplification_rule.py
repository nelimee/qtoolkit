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

"""Implements the SimplificationRule class."""

import math
import typing

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as gates
from qtoolkit.data_structures.quantum_circuit.simplifier.gate_parameter import \
    GateParameter

GateIdentifier = str


class SimplificationRule:
    """Representation of a generic simplification rule."""

    def __init__(self, rule: typing.List[GateIdentifier],
                 parameters: typing.List[
                     typing.Optional[GateParameter]]) -> None:
        """Initialise the SimplificationRule class.

        :param rule: each string in the list represents a quantum gate. The
        sequence represented by this list of strings is "simplifiable".
        :param parameters: a list of optional parameters. If rule[i] represents
        a parametrised quantum gate then parameters[i] represents a rule that
        needs to be checked for the sequence to be simplifiable. Each parameter
        has an identifier and a transformation function. The real value of a
        parameter is obtained by calling the transformation function on the
        corresponding gate parameters. For a rule to be fulfilled, all the
        parameters with the same identifier should have the same real value
        (up to machine precision).
        """
        self._rule = rule
        self._has_parameters = any(parameters)
        self._parameters = parameters

    def is_simplifiable(self, quantum_gate_sequence: typing.List[
        gates.QuantumGate]) -> bool:
        """Check if the given sequence is simplifiable.

        :param quantum_gate_sequence: the sequence of quantum gates to check for
        simplifiability.
        :return: True if the sequence is simplifiable according to the rule
        stored, else False.
        """
        # If there are not enough gates in the sequence to apply the rule then
        # the sequence is not simplifiable.
        if len(quantum_gate_sequence) < len(self._rule):
            return False

        # We try to find a subsequence in the given sequence that may be
        # simplifiable.
        sequence_str = ''.join([gate.name for gate in quantum_gate_sequence])
        rule_str = ''.join(self._rule)
        position = sequence_str.find(rule_str)
        # If there is no sequence that match, we are done.
        if position == -1:
            return False

        # Else, there is at least one sequence. If the gates in the rule don't
        # have parameters, then we can directly return True.
        elif not self._has_parameters:
            return True

        # Else, while there is a potentially simplifiable sequence we need to
        # check if the parameters make this sequence simplifiable.
        while position != -1:
            sequence = quantum_gate_sequence[
                       position:position + len(self._rule)]
            if self.is_simplifiable_exact_length(sequence):
                return True
            position = sequence_str.find(rule_str, position)

    def is_simplifiable_from_last(self, quantum_gate_sequence: typing.List[
        gates.QuantumGate]) -> bool:
        """Check if the last part of the given sequence is simplifiable.

        This method can be used to check if the last gate of the sequence
        introduced a possible simplification or not.

        :param quantum_gate_sequence: the sequence of quantum gates to check for
        simplifiability.
        :return: True if the sequence is simplifiable according to the rule
        stored, else False.
        """
        # If there are not enough gates in the sequence to apply the rule then
        # the sequence is not simplifiable.
        if len(quantum_gate_sequence) < len(self._rule):
            return False

        return self.is_simplifiable_exact_length(
            quantum_gate_sequence[-len(self._rule):])

    def is_simplifiable_exact_length(self, quantum_gate_sequence: typing.List[
        gates.QuantumGate]) -> bool:
        """Check if the given sequence is simplifiable.

        The given sequence needs to have exactly the same length as the rule
        represented by the instance this method is called on.

        :param quantum_gate_sequence: the sequence of quantum gates to check for
        simplifiability.
        :return: True if the sequence is simplifiable according to the rule
        stored, else False.
        """

        assert len(quantum_gate_sequence) == len(self._rule)

        # Check the gates identifiers
        for idx, gate in enumerate(quantum_gate_sequence):
            if gate.name != self._rule[idx]:
                return False

        # Check the gates parameters
        parameters = dict()
        for idx, gate in enumerate(quantum_gate_sequence):
            is_parametrised = (gate.parameters is not None and self._parameters[
                idx] is not None)
            if not is_parametrised:
                continue
            # Recover the parameter's data.
            parameter_ID = self._parameters[idx].id
            parameter_value = self._parameters[idx].apply_transformation(
                gate.parameters[0])
            # Check the parameter value.
            if parameter_ID not in parameters:
                parameters[parameter_ID] = parameter_value
            elif not math.isclose(parameters[parameter_ID], parameter_value):
                return False

        return True
