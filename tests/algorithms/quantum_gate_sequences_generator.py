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

"""Test the quantum gate sequence generator."""

import copy
import typing
import unittest

import numpy

import qtoolkit.utils.constants.matrices as mconsts
import qtoolkit.utils.types as qtypes
import tests.qtestcase as qtest
from qtoolkit.algorithms.quantum_gate_sequences_generator import \
    generate_all_gate_sequences


class QuantumGateSequencesGeneratorTestCase(qtest.QTestCase):
    """Unit-tests for the sequences generation function."""

    @staticmethod
    def _perform_gen(*args, **kwargs) -> typing.List[
        typing.Tuple[numpy.ndarray, qtypes.SUdMatrix]]:
        return list(
            map(copy.deepcopy, generate_all_gate_sequences(*args, **kwargs)))

    @classmethod
    def setUpClass(cls):
        """Setup only once for constants to avoid useless computations."""
        cls._basis = [mconsts.H_SU2, mconsts.H_SU2.T.conj(), mconsts.T_SU2,
                      mconsts.T_SU2.T.conj()]
        cls._simplifications = {(0, 1), (1, 0), (2, 2, 3, 3), (3, 3, 2, 2),
                                (2, 3), (3, 2)}
        cls._max_simplification_length = 4

        cls._small_basis = [mconsts.H, mconsts.T, mconsts.T.T.conj()]
        cls._small_simplifications = {(0, 0), (1, 2), (2, 1), (2, 2, 1, 1),
                                      (1, 1, 2, 2)}
        cls._small_max_simplification_length = 4

    def test_negative_depth(self) -> None:
        """Negative depth."""
        with self.assertRaises(AssertionError):
            self._perform_gen(self._basis, -1, self._simplifications)

        with self.assertRaises(AssertionError):
            self._perform_gen(self._basis, 0, self._simplifications)

    def test_empty_basis(self) -> None:
        """The given basis is empty."""
        with self.assertRaises(AssertionError):
            self._perform_gen([], 0, self._simplifications)

    def test_too_large_basis(self) -> None:
        """The given basis is too large."""
        # Should not raise anything.
        self._perform_gen([mconsts.H_SU2] * 255, 1, self._simplifications)
        self._perform_gen([mconsts.H_SU2] * 65535, 1, self._simplifications,
                          gate_id_type=numpy.uint16)
        with self.assertRaises(AssertionError):
            self._perform_gen([mconsts.H_SU2] * 256, 1, self._simplifications)
        with self.assertRaises(AssertionError):
            self._perform_gen([mconsts.H_SU2] * 65536, 1, self._simplifications,
                              gate_id_type=numpy.uint16)

    def test_simplification_not_in_basis(self) -> None:
        """A gate in the simplifications is not in the basis."""
        with self.assertRaises(AssertionError):
            self._perform_gen([mconsts.H_SU2, mconsts.H_SU2.T.conj()], 1,
                              {(0, 1), (2, 3)})

    def test_generation_depth_1_no_nodes(self) -> None:
        """Generated gates without nodes at depth 1 are correct."""
        generated_gates = self._perform_gen(self._basis, 1,
                                            self._simplifications)
        # The generated gates should exactly match the given basis, except for
        # the order that can change.
        self.assertEqual(len(generated_gates), len(self._basis),
                         f"The number of generated sequences of depth 1 "
                         f"({len(generated_gates)}) mismatch with the expected "
                         f"number ({len(self._basis)}).")

        matched_gates = numpy.zeros((len(self._basis)), dtype=bool)
        for idx, basis_gate in enumerate(self._basis):
            for gen_gate in generated_gates:
                if numpy.allclose(basis_gate, gen_gate[1]):
                    matched_gates[idx] = True
                    break
        self.assertTrue(numpy.all(matched_gates),
                        "The generated gates and the expected ones mismatch.")

    def test_generation_depth_1_with_nodes(self) -> None:
        """Generated gates with nodes at depth 1 are correct."""
        generated_gates = self._perform_gen(self._basis, 1,
                                            self._simplifications,
                                            include_nodes=True)
        # The generated gates should exactly match the given basis, except for
        # the order that can change.
        self.assertEqual(len(generated_gates), len(self._basis),
                         f"The number of generated sequences of depth 1 "
                         f"({len(generated_gates)}) mismatch with the expected "
                         f"number ({len(self._basis)}).")

        matched_gates = numpy.zeros((len(self._basis)), dtype=bool)
        for idx, basis_gate in enumerate(self._basis):
            for gen_gate in generated_gates:
                if numpy.allclose(basis_gate, gen_gate[1]):
                    matched_gates[idx] = True
                    break
        self.assertTrue(numpy.all(matched_gates),
                        "The generated gates and the expected ones mismatch.")

    def test_generation_depth_2_no_nodes(self) -> None:
        """Generated gates without nodes at depth 2 are correct."""
        generated_gates = self._perform_gen(self._small_basis, 2,
                                            self._small_simplifications)
        H, T, Tinv = mconsts.H, mconsts.T, mconsts.T.T.conj()
        expected_results = [(numpy.array([0, 1], dtype=numpy.uint8), H @ T),
                            (numpy.array([0, 2], dtype=numpy.uint8), H @ Tinv),
                            (numpy.array([1, 0], dtype=numpy.uint8), T @ H),
                            (numpy.array([1, 1], dtype=numpy.uint8), T @ T),
                            (numpy.array([2, 0], dtype=numpy.uint8), Tinv @ H),
                            (numpy.array([2, 2], dtype=numpy.uint8),
                             Tinv @ Tinv), ]
        self.assertEqual(len(generated_gates), len(expected_results),
                         f"The number of generated sequences of depth 2 "
                         f"({len(generated_gates)}) mismatch with the expected "
                         f"number ({len(expected_results)}).")

        matched_gates = numpy.zeros((len(expected_results)), dtype=bool)
        for idx, result in enumerate(expected_results):
            for gen in generated_gates:
                if numpy.all(gen[0] == result[0]) and numpy.allclose(gen[1],
                                                                     result[1]):
                    matched_gates[idx] = True
                    break
        self.assertTrue(numpy.all(matched_gates),
                        "The generated gates and the expected ones mismatch.")

    def test_generation_depth_2_with_nodes(self) -> None:
        """Generated gates with nodes at depth 2 are correct."""
        generated_gates = self._perform_gen(self._small_basis, 2,
                                            self._small_simplifications,
                                            include_nodes=True)
        H, T, Tinv = mconsts.H, mconsts.T, mconsts.T.T.conj()
        expected_results = [(numpy.array([0], dtype=numpy.uint8), H),
                            (numpy.array([0, 1], dtype=numpy.uint8), H @ T),
                            (numpy.array([0, 2], dtype=numpy.uint8), H @ Tinv),
                            (numpy.array([1], dtype=numpy.uint8), T),
                            (numpy.array([1, 0], dtype=numpy.uint8), T @ H),
                            (numpy.array([1, 1], dtype=numpy.uint8), T @ T),
                            (numpy.array([2], dtype=numpy.uint8), Tinv),
                            (numpy.array([2, 0], dtype=numpy.uint8), Tinv @ H),
                            (numpy.array([2, 2], dtype=numpy.uint8),
                             Tinv @ Tinv), ]
        self.assertEqual(len(generated_gates), len(expected_results),
                         f"The number of generated sequences of depth 2 "
                         f"({len(generated_gates)}) mismatch with the expected "
                         f"number ({len(expected_results)}).")

        matched_gates = numpy.zeros((len(expected_results)), dtype=bool)
        for idx, result in enumerate(expected_results):
            for gen in generated_gates:
                if gen[0].shape == result[0].shape and numpy.all(
                    gen[0] == result[0]) and numpy.allclose(gen[1], result[1]):
                    matched_gates[idx] = True
                    break
        self.assertTrue(numpy.all(matched_gates),
                        "The generated gates and the expected ones mismatch.")


if __name__ == '__main__':
    unittest.main()
