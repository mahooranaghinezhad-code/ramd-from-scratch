"""Numerical/logic regression tests for independent checkpoint randomization."""
from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.feedback_vs_exponential import Config, simulate, randomized_checkpoint_mask
from research.checkpoint_randomized_control import Plan, config_for, run, summarize


class CheckpointRandomControlTests(unittest.TestCase):
    def test_random_subset_honors_exact_quota(self):
        rng=np.random.default_rng(17)
        seen=0
        reset=0
        for n in (13,9,21,5,18,7):
            mask=randomized_checkpoint_mask(n,.37,seen,reset,rng)
            self.assertEqual(len(mask),n)
            reset+=int(mask.sum());seen+=n
            self.assertEqual(reset,int(np.floor(.37*seen+.5)))
        self.assertLess(abs(reset-.37*seen),.51)

    def test_random_subset_is_not_the_deterministic_progress_rule(self):
        # With the SAME random seed but differently ordered hypothetical
        # displacements, the selector takes only the count and p, not progress.
        a=randomized_checkpoint_mask(100,.5,0,0,np.random.default_rng(7))
        b=randomized_checkpoint_mask(100,.5,0,0,np.random.default_rng(7))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(int(a.sum()),50)
        c=randomized_checkpoint_mask(100,.5,0,0,np.random.default_rng(8))
        self.assertFalse(np.array_equal(a,c))

    def test_invalid_checkpoint_probability(self):
        for p in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                randomized_checkpoint_mask(20,p,0,0,np.random.default_rng(1))
            with self.assertRaises(ValueError):
                simulate(Config(replicas=8,steps=20), 'checkpoint_random',seed=5,
                         switch_probability=p)
        with self.assertRaises(ValueError):
            simulate(Config(replicas=8,steps=20),'checkpoint_random',seed=5)

    def test_randomized_switch_frequency_is_independent_of_displacement(self):
        cfg=Config(replicas=48,steps=700,check_steps=20,dt=.005)
        r=simulate(cfg,'checkpoint_random',seed=6,switch_probability=.31)
        self.assertGreater(r['checkpoint_opportunities'],0)
        self.assertLessEqual(abs(r['n_resets']-.31*r['checkpoint_opportunities']),.50000001)
        self.assertAlmostEqual(r['n_resets']/r['checkpoint_opportunities'],
                               r['checkpoint_reset_fraction'])
        self.assertLessEqual(r['n_escaped']+r['n_censored'],cfg.replicas)
        self.assertEqual(r['n_escaped']+r['n_censored'],cfg.replicas)

    def test_p_zero_and_p_one(self):
        cfg=Config(replicas=20,steps=60,check_steps=10)
        z=simulate(cfg,'checkpoint_random',seed=4,switch_probability=0.)
        one=simulate(cfg,'checkpoint_random',seed=4,switch_probability=1.)
        self.assertEqual(z['n_resets'],0)
        self.assertEqual(one['n_resets'],one['checkpoint_opportunities'])

    def test_old_feedback_mode_records_checkpoint_decisions(self):
        cfg=Config(replicas=36,steps=200,check_steps=10)
        fb=simulate(cfg,'feedback',seed=8)
        self.assertGreater(fb['checkpoint_opportunities'],0)
        self.assertLessEqual(fb['n_resets'],fb['checkpoint_opportunities'])
        self.assertAlmostEqual(fb['n_resets']/fb['checkpoint_opportunities'],
                               fb['checkpoint_reset_fraction'])

    def test_complete_small_three_protocol_grid(self):
        plan=Plan(forces=(1.,),thresholds=(.55,),seeds=(2026,2027),
                  replicas=48,horizon=8.)
        with tempfile.TemporaryDirectory() as tmp:
            rows,aggregate,ctrl=run(plan,Path(tmp))
            self.assertEqual(len(rows),2)
            self.assertEqual(len(aggregate),1)
            self.assertEqual(len(ctrl),2)
            for row in rows:
                self.assertEqual(row['status'],'valid')
                self.assertEqual(row['rand_quota_error'],0)
                self.assertTrue(np.isfinite(row['delta_fb_rand_rmst']))
                self.assertGreater(row['n'],0)
                self.assertGreater(row['rand_eligible'],0)
            for f in ('per_seed.csv','conditions.csv','unbiased_control.csv',
                      'checkpoint_randomized_comparison.png','LIMITATIONS.txt'):
                self.assertTrue((Path(tmp)/f).exists(),f)
            with (Path(tmp)/'per_seed.csv').open(newline='') as fp:
                self.assertEqual(len(list(csv.DictReader(fp))),2)


if __name__=='__main__':
    unittest.main()
