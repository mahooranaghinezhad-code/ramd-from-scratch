"""Fast regression checks for the sensitivity-study protocol controls."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from research.feedback_sensitivity_scan import (
    ScanPlan, aggregate, ci95_student, config_for, run, scan, validate,
)


class SweepTests(unittest.TestCase):
    def test_fixed_physical_time_across_timestep(self):
        plan = ScanPlan(timesteps=(0.005, 0.0025), replicas=8, horizon=10)
        for dt in plan.timesteps:
            cfg = config_for(plan, force=2.0, threshold=0.5, dt=dt)
            self.assertAlmostEqual(cfg.dt*cfg.steps, 10.0)
            self.assertAlmostEqual(cfg.dt*cfg.check_steps, 0.1)
            self.assertAlmostEqual(cfg.dt*cfg.trace_stride, 0.025)

    def test_student_ci(self):
        self.assertEqual(ci95_student([2, 2, 2]), (2.0, 2.0, 2.0))
        mean, lo, hi = ci95_student([-1, 0, 1])
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(-lo, hi)
        self.assertTrue(lo < 0 < hi)

    def test_invalid_and_duplicate_inputs(self):
        with self.assertRaises(ValueError):
            validate(ScanPlan(timesteps=(0.0,)))
        with self.assertRaises(ValueError):
            validate(ScanPlan(seeds=(2026, 2026)))
        with self.assertRaises(ValueError):
            validate(ScanPlan(replicas=1))

    def test_no_feedback_resets_are_not_imputed(self):
        plan=ScanPlan(forces=(1.,),thresholds=(0.,),timesteps=(.005,),
                      seeds=(2026,2027),replicas=8,horizon=.1)
        rows, controls=scan(plan)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['status']=='no_feedback_resets' for r in rows))
        self.assertEqual(aggregate(rows)[0]['n_valid_batches'], 0)
        self.assertEqual(len(controls),2)

    def test_small_scan_writes_all_outputs(self):
        plan=ScanPlan(forces=(1.,),thresholds=(.55,),timesteps=(.005,.0025),
                      seeds=(2035,2036),replicas=24,horizon=4.)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            rows, agg, controls=run(plan,out)
            self.assertEqual(len(rows),4)
            self.assertEqual(len(agg),2)
            self.assertEqual(len(controls),4)
            self.assertTrue(all(r['status']=='valid' for r in rows))
            self.assertTrue(all(np.isfinite(r['delta_rmst']) for r in rows))
            for name in ['per_seed.csv','conditions.csv','unbiased_control.csv',
                         'sensitivity_scan.png','LIMITATIONS.txt']:
                self.assertTrue((out/name).is_file(),name)


if __name__ == '__main__':
    unittest.main()
