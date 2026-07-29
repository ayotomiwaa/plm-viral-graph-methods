#!/usr/bin/env python3
"""Synthetic, sequence-free checks for paired_kmedoids_comparison.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validation/paired_kmedoids_comparison.py"
SPEC = importlib.util.spec_from_file_location("paired_kmedoids_comparison_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {SCRIPT}")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class PairedKMedoidsTests(unittest.TestCase):
    def test_profile_medoid_equals_dense_hamming_medoid(self) -> None:
        rng = np.random.default_rng(123)
        for n_members in (2, 7, 25, 100):
            encoded = rng.integers(0, 8, size=(n_members, 41), dtype=np.uint8)
            members = np.arange(n_members, dtype=np.int64)
            observed = MOD.profile_hamming_medoid(encoded, members, block_size=13)
            dense = (encoded[:, None, :] != encoded[None, :, :]).sum(axis=2)
            scores = dense.sum(axis=1)
            expected = int(np.flatnonzero(scores == scores.min()).min())
            self.assertEqual(observed, expected)

    def test_dense_assignment_preserves_medoid_ownership_on_zero_ties(self) -> None:
        dense = np.array(
            [
                [65535.0, 0.0, 4.0],
                [0.0, 65535.0, 4.0],
                [4.0, 4.0, 65535.0],
            ]
        )
        labels, objective = MOD.assign_dense(dense, np.array([0, 1], dtype=np.int64))
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[1], 1)
        self.assertEqual(objective, 4.0)

    def test_all_node_rng_update_matches_dense_exact_update(self) -> None:
        coordinates = np.array([0.0, 0.5, 1.5, 7.0, 8.0, 10.0])
        dense = np.abs(coordinates[:, None] - coordinates[None, :])
        medoids = np.array([0, 5], dtype=np.int64)
        labels, _ = MOD.assign_dense(dense, medoids)
        expected = MOD.update_dense_exact(
            dense,
            labels,
            k=2,
            old_medoids=medoids,
            candidate_block_size=2,
            member_block_size=3,
        )
        candidate_node_ids = np.arange(dense.shape[0], dtype=np.int64)
        observed = MOD.update_candidate_rows(
            dense,
            labels,
            k=2,
            old_medoids=medoids,
            candidate_node_ids=candidate_node_ids,
            node_to_candidate_row={int(x): int(x) for x in candidate_node_ids},
            member_block_size=3,
        )
        np.testing.assert_array_equal(observed, expected)

    def test_external_evaluation_is_permutation_invariant(self) -> None:
        truth = np.repeat(np.arange(3), 4)
        predicted = np.array([2] * 4 + [0] * 4 + [1] * 4)
        report = MOD.evaluate_labels(predicted, truth, ["a", "b", "c"], k=3)
        self.assertEqual(report["ari"], 1.0)
        self.assertEqual(report["n_mislabeled"], 0)

    def test_candidate_fingerprint_is_order_sensitive(self) -> None:
        candidates = pd.DataFrame({"candidate_row": [0, 1, 2], "node_id": [4, 7, 9]})
        reordered = pd.DataFrame({"candidate_row": [0, 1, 2], "node_id": [4, 9, 7]})
        self.assertNotEqual(
            MOD.candidate_fingerprint(candidates),
            MOD.candidate_fingerprint(reordered),
        )


if __name__ == "__main__":
    unittest.main()
