from __future__ import annotations

import unittest

from ep_predict.runtime import _base_architecture


class RuntimeHelpersTest(unittest.TestCase):
    def test_base_architecture_strips_rocm_feature_suffixes(self) -> None:
        self.assertEqual(
            _base_architecture("gfx950:sramecc+:xnack-"),
            "gfx950",
        )

    def test_base_architecture_preserves_plain_name(self) -> None:
        self.assertEqual(_base_architecture("gfx950"), "gfx950")
        self.assertIsNone(_base_architecture(None))


if __name__ == "__main__":
    unittest.main()
