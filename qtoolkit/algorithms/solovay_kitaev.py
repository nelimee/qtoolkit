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

import qtoolkit.data_structures.nearest_neighbour_structure as nn_struct
import qtoolkit.data_structures.quantum_gate_sequence as qgate_seq
import qtoolkit.maths.matrix.su2.group_commutator as su2_gc
import qtoolkit.maths.matrix.su2.transformations as su2_trans
import qtoolkit.maths.matrix.sud.group_commutator as sud_gc
import qtoolkit.utils.types as qtypes


def solovay_kitaev(U: qtypes.SUdMatrix, recursion_level: int,
                   approximations: qtypes.NearestNeighbourQueryable) -> \
    qgate_seq.QuantumGateSequence:
    """Implementation of the Solovay-Kitaev theorem for SU(d) matrices.

    The implementation follows https://github.com/cmdawson/sk.

    :param U: The SU(d) matrix to decompose.
    :param recursion_level: The number of recursive calls.
    :param approximations: An efficient structure to perform nearest-neighbour
    queries on SU(d) matrices. The structure needs to return an instance of the
    QuantumGateSequence class.
    :return: a sequence of quantum gates approximating the given SU(d) matrix U.
    """
    if recursion_level == 0:
        _, approximation = approximations.query(U)
        return approximation
    Un_1 = solovay_kitaev(U, recursion_level - 1, approximations)
    V, W = sud_gc.group_commutator(U @ Un_1.matrix.T.conj())
    Vn_1 = solovay_kitaev(V, recursion_level - 1, approximations)
    Wn_1 = solovay_kitaev(W, recursion_level - 1, approximations)
    return Vn_1 @ Wn_1 @ Vn_1.inverse() @ Wn_1.inverse() @ Un_1


def solovay_kitaev_su2(U: qtypes.SU2Matrix, recursion_level: int,
                       approximations: nn_struct.NearestNeighbourStructure) \
    -> qgate_seq.QuantumGateSequence:
    """Implementation of the Solovay-Kitaev theorem for SU(2) matrices.

    The implementation follows https://github.com/cmdawson/sk.

    :param U: The SU(2) matrix to decompose.
    :param recursion_level: The number of recursive calls.
    :param approximations: An efficient structure to perform nearest-neighbour
    queries on SO(3) vectors.
    :return: a sequence of quantum gates approximating the given SU(2) matrix U.
    """
    if recursion_level == 0:
        _, approx_qgate_seq = approximations.query(su2_trans.su2_to_so3(U))
        return approx_qgate_seq
    Un_1 = solovay_kitaev_su2(U, recursion_level - 1, approximations)
    V, W = su2_gc.su2_group_commutator_decompose(U @ Un_1.matrix.T.conj())
    Vn_1 = solovay_kitaev_su2(V, recursion_level - 1, approximations)
    Wn_1 = solovay_kitaev_su2(W, recursion_level - 1, approximations)
    return Vn_1 @ Wn_1 @ Vn_1.inverse() @ Wn_1.inverse() @ Un_1
