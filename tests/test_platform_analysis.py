from __future__ import annotations

import unittest

from ep_predict.analysis.platform import _pearson, _ranks, _spearman


class PlatformAnalysisTest(unittest.TestCase):
    def test_correlations_preserve_and_reverse_trends(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        self.assertAlmostEqual(_pearson(values, values), 1.0)
        self.assertAlmostEqual(_spearman(values, list(reversed(values))), -1.0)

    def test_ranks_average_ties(self) -> None:
        self.assertEqual(_ranks([4.0, 1.0, 1.0, 3.0]), [4.0, 1.5, 1.5, 3.0])


if __name__ == "__main__":
    unittest.main()
