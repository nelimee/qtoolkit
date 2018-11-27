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

"""Frequently used quantum gates."""

import qtoolkit.data_structures.quantum_circuit.gate_hierarchy as qgates
import qtoolkit.utils.constants.matrices as mconsts


def _self_inverse(gate: qgates.QuantumGate) -> qgates.QuantumGate:
    return gate


def _generic_inverse(gate: qgates.QuantumGate) -> qgates.QuantumGate:
    name = (gate.name + '+').replace('++', '')
    return qgates.QuantumGate(name, gate.matrix.T.conj(), _generic_inverse,
                              parameters=gate.parameters)


def _inverse_angle(gate: qgates.QuantumGate) -> qgates.QuantumGate:
    return qgates.QuantumGate(gate.name, gate.matrix.T.conj(), _inverse_angle,
                              parameters=-gate.parameters)


X = qgates.QuantumGate('X', mconsts.X, _self_inverse)
Y = qgates.QuantumGate('Y', mconsts.Y, _self_inverse)
Z = qgates.QuantumGate('Z', mconsts.Z, _self_inverse)
H = qgates.QuantumGate('H', mconsts.H, _self_inverse)
S = qgates.QuantumGate('S', mconsts.S, _generic_inverse)
T = qgates.QuantumGate('T', mconsts.T, _generic_inverse)
ID = qgates.QuantumGate('Id', mconsts.ID2, _self_inverse)
CX = qgates.QuantumGate('CX', mconsts.CX, _self_inverse)

Rx = qgates.ParametrisedQuantumGate('Rx', mconsts.Rx, _inverse_angle)
Ry = qgates.ParametrisedQuantumGate('Ry', mconsts.Ry, _inverse_angle)
Rz = qgates.ParametrisedQuantumGate('Rz', mconsts.Rz, _inverse_angle)
