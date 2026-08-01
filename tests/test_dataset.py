from __future__ import annotations

import unittest

from ep_predict.data.standard import (
    _extract_source_value,
    _normalized_text,
    _source_sample_id,
)


class DatasetPreparationTest(unittest.TestCase):
    def test_normalizes_prose(self) -> None:
        self.assertEqual(_normalized_text("  one\n  two  "), "one two")

    def test_preserves_code_whitespace(self) -> None:
        source = "def f():\n    return 1\n"
        self.assertEqual(
            _normalized_text(source, preserve_whitespace=True),
            "def f():\n    return 1",
        )

    def test_stable_safe_source_id(self) -> None:
        source = {"key": "humaneval", "id_field": "task_id"}
        self.assertEqual(
            _source_sample_id(source, {"task_id": "HumanEval/17"}, 0),
            "humaneval-HumanEval_17",
        )

    def test_extracts_first_conversation_turn(self) -> None:
        self.assertEqual(
            _extract_source_value(
                {"field": "prompt", "value_mode": "first"},
                {"prompt": ["first", "second"]},
            ),
            "first",
        )


if __name__ == "__main__":
    unittest.main()
