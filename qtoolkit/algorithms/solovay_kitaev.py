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

"""Implementation of the Solovay-Kitaev algorithm.

The implementation is largely inspired from 2 GitHub repositories:

1. `The implementation by Chris Dawson <https://github.com/cmdawson/sk>`_.
2. `A Python 2 implementation Paul Pham \
    <https://github.com/cryptogoth/skc-python>`_
"""

import qtoolkit.data_structures.nearest_neighbour_structure as qnn
import qtoolkit.data_structures.quantum_circuit.quantum_circuit as qcirc
import qtoolkit.maths.matrix.sud.group_commutator as sud_gc
import qtoolkit.utils.types as qtypes


def solovay_kitaev(
    unitary: qtypes.UnitaryMatrix,
    recursion_level: int,
    approximations: qnn.NearestNeighbourStructure,
) -> qcirc.QuantumCircuit:
    """Implementation of the Solovay-Kitaev algorithm for :math:`U(2^n)`.

    :param unitary: The unitary matrix to decompose. `unitary` should be in
        :math:`U(2^n)` (the group of all the :math:`2^n\times 2^n` unitary
        matrices).
    :param recursion_level: The number of recursive calls to perform. Precision
        of the result decrease exponentially with the value of
        `recursion_level`.
    :param approximations: An efficient structure to perform nearest-neighbour
        queries on unitary matrices. The structure needs to return a tuple
        (distance, circuit) of type (int, :py:class:`~.QuantumCircuit`).
    :return: a sequence of quantum gates approximating `unitary`.
    """
    if recursion_level == 0:
        _, approximation = approximations.query(unitary)
        return approximation
    Un_1 = solovay_kitaev(unitary, recursion_level - 1, approximations)
    V, W = sud_gc.group_commutator(unitary @ Un_1.matrix.T.conj())
    Vn_1 = solovay_kitaev(V, recursion_level - 1, approximations)
    Wn_1 = solovay_kitaev(W, recursion_level - 1, approximations)
    return Vn_1 @ Wn_1 @ Vn_1.inverse() @ Wn_1.inverse() @ Un_1
