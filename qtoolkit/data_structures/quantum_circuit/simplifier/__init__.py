# ======================================================================
# Copyright CERFACS (November 2018)
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

"""Simplification detection and modification for :py:class:`~.QuantumCircuit`.

This package aims at implementing reusable classes/methods to detect and
"correct" simplifications in a :py:class:`~.QuantumCircuit`.

Definitions:

# In the whole package, a :py:class:`~.QuantumCircuit` is said to be
  "simplifiable" if it admits at least one sub-sequence :math:`S` checking at
  least one of the following points:

  * The :py:class:`~.QuantumCircuit` represented by :math:`S` acts on a quantum
    state as the identity matrix.
  * The matrix :math:`M` representing :math:`S` represents also a shorter (in
    terms of quantum gate number) sequence of quantum gates :math:`S'`.


"""
