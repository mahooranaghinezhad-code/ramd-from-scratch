"""Deterministic unit / smoke controls for the protocol-comparison toy."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from research.feedback_vs_exponential import (
    Config, exact_unbiased_mfpt, feedback_switch, run, simulate,
)


class FeedbackProtocolTests(unittest.TestCase):
    def test_backward_equation_flat_well(self):
        cfg = Config(height=0.0)
        analytic = (cfg.right - cfg.start) * (cfg.right + cfg.start - 2*cfg.left) / (2*cfg.kbt)
        self.assertAlmostEqual(exact_unbiased_mfpt(cfg), analytic, delta=1e-3)

    def test_quadrature_refinement(self):
        cfg = Config()
        coarse = exact_unbiased_mfpt(cfg, n_grid=1601)
        fine = exact_unbiased_mfpt(cfg, n_grid=16001)
        self.assertLess(abs(coarse-fine), 0.003)

    def test_switch_rule(self):
        np.testing.assert_array_equal(
            feedback_switch(np.array([0.1, 0.42, 0.7]), 0.42), [True, False, False]
        )

    def test_short_paths_are_censored(self):
        r = simulate(Config(replicas=12, steps=1), "feedback", seed=4)
        self.assertEqual(r["n_censored"], 12)
        self.assertIsNone(r["mfpt_if_complete"])
        self.assertAlmostEqual(r["rmst"], 0.005)

    def test_invalid_exponential_tau(self):
        with self.assertRaises(ValueError):
            simulate(Config(replicas=8,steps=10), "exponential", seed=3, tau=0)

    def test_exact_reference_matches_unbiased_statistical_control(self):
        cfg = Config(replicas=384, steps=14000)
        r = simulate(cfg, "unbiased", seed=2030)
        self.assertGreater(r["n_escaped"], 360)
        exact = exact_unbiased_mfpt(cfg)
        self.assertLess(abs(r["rmst"] - exact) / exact, 0.28)

    def test_matched_protocol_outputs(self):
        cfg = Config(replicas=96, steps=4500)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            rows=run(cfg,seed=2026,output=output)
            self.assertEqual([r["protocol"] for r in rows],
                             ["feedback","exponential","unbiased"])
            fb_tau=rows[0]["mean_orientation_time_per_reset"]
            ex_tau=rows[1]["mean_orientation_time_per_reset"]
            self.assertIsNotNone(fb_tau)
            self.assertIsNotNone(ex_tau)
            self.assertLess(abs(fb_tau-ex_tau)/fb_tau, 0.25)
            self.assertTrue(np.isfinite(rows[0]["persistence_ratio_out_in"]))
            for filename in ("summary.csv","escape_times.csv","orientation_correlation.csv",
                             "comparison.png","README.txt"):
                self.assertTrue((output/filename).exists(),filename)
            with (output/"summary.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),3)
            data=np.genfromtxt(output/"orientation_correlation.csv",delimiter=",",names=True)
            self.assertAlmostEqual(float(data["exponential"][0]),1.0,delta=0.0001)


if __name__ == "__main__":
    unittest.main()
