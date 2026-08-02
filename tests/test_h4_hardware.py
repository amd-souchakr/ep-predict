from __future__ import annotations

import unittest

from ep_predict.hardware.h4 import _validate_hardware_report


class H4HardwareTest(unittest.TestCase):
    def test_accepts_frozen_mi355x_scope(self) -> None:
        _validate_hardware_report(
            {
                "torch_backend": "rocm",
                "gpu": {
                    "device_count": 1,
                    "architecture": "gfx950:sramecc+:xnack-",
                },
            },
            {"visible_device_count": 1, "required_architecture": "gfx950"},
        )

    def test_rejects_non_rocm_backend(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ROCm"):
            _validate_hardware_report(
                {
                    "torch_backend": "cuda",
                    "gpu": {"device_count": 1, "architecture": None},
                },
                {"visible_device_count": 1, "required_architecture": "gfx950"},
            )


if __name__ == "__main__":
    unittest.main()
