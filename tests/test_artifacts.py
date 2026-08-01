from __future__ import annotations

import unittest
from pathlib import Path

from ep_predict.artifacts import _is_disposable, _manifest_entries


class ArtifactRetentionTest(unittest.TestCase):
    def test_disposable_directories_are_path_components(self) -> None:
        root = Path("/repo/artifacts")
        self.assertTrue(
            _is_disposable(
                root / "runs" / "run-1" / "trace" / "request.jsonl.gz",
                root,
            )
        )
        self.assertTrue(
            _is_disposable(
                root / "runs" / "run-1" / "features" / "request.npz",
                root,
            )
        )
        self.assertFalse(
            _is_disposable(
                root / "runs" / "run-1" / "analysis" / "trace_integrity.json",
                root,
            )
        )

    def test_manifest_entries_support_all_existing_shapes(self) -> None:
        list_shape = [{"path": "a.csv", "sha256": "aaa"}]
        labeled_shape = {
            "summary": {"path": "summary.csv", "sha256": "bbb"}
        }
        direct_shape = {"figure.pdf": "ccc"}

        self.assertEqual(list(_manifest_entries(list_shape)), [("a.csv", "aaa")])
        self.assertEqual(
            list(_manifest_entries(labeled_shape)),
            [("summary.csv", "bbb")],
        )
        self.assertEqual(
            list(_manifest_entries(direct_shape)),
            [("figure.pdf", "ccc")],
        )


if __name__ == "__main__":
    unittest.main()
