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

"""Test of the GateHierarchy class."""
import unittest

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgates
import qtoolkit.utils.constants.matrices as mconsts
import tests.qtestcase as qtest


class GateHierarchyTestCase(qtest.QTestCase):
    """Unit-tests for the GateHierarchy class."""

    def test_creation_instruction(self) -> None:
        """Test if construction of QuantumInstruction works."""
        qinst = qgates.QuantumInstruction("qinst")

    def test_creation_gate_no_parameters(self) -> None:
        """Test if construction of QuantumGate without parameter works."""
        qgate = qgates.QuantumGate("qgate", mconsts.X)

    def test_creation_gate_parameters(self) -> None:
        """Test if construction of QuantumGate with parameter(s) works."""
        # TODO
        pass

    def test_creation_parametrised_gate(self) -> None:
        """Test if construction of ParametrisedQuantumGate works."""
        # TODO
        pass

    def test_instruction_name(self) -> None:
        """Test if QuantumInstruction's name attribute is right."""
        qinst = qgates.QuantumInstruction("qinst")
        self.assertEqual(qinst.name, "qinst")

    def test_gate_matrix(self) -> None:
        """Test if QuantumGate's matrix attribute is right."""
        qgate = qgates.QuantumGate("qgate", mconsts.X)
        self.assertAllClose(qgate.matrix, mconsts.X)

    def test_gate_parameter_empty(self) -> None:
        """Test if QuantumGate's parameters attribute is right."""
        qgate = qgates.QuantumGate("qgate", mconsts.X)
        self.assertFalse(qgate.parameters)

    def test_gate_name(self) -> None:
        """Test if QuantumGate's name attribute is right."""
        qgate = qgates.QuantumGate("qgate", mconsts.X)
        self.assertEqual(qgate.name, "qgate")

    def test_gate_dim(self) -> None:
        """Test if QuantumGate's dim attribute is right."""
        qgate = qgates.QuantumGate("qgate", mconsts.X)
        self.assertEqual(qgate.dim, 2)

    def test_parametrised_gate_name(self) -> None:
        """Test if ParametrisedQuantumGate's name attribute is right."""
        # TODO
        pass

    def test_parametrised_gate_call(self) -> None:
        """Test if ParametrisedQuantumGate's __call__ is right."""
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
