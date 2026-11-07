"""Regression tests for frozen classifier transfer and MFPT reconstruction.

These tests verify *design and numerical integrity*, not physical validity.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from research.feedback_vs_exponential import Config, exact_unbiased_mfpt
from research.heldout_kinetic_reliability import (
    Plan, validate, cfg_for, effective_alpha, kinetic_extrapolation,
    spearman, rank, seed_for, evaluate,
)
from research.state_conditioned_switching import RidgeLogistic


class HeldoutReliabilityTests(unittest.TestCase):
    def test_heldout_physics_and_seed_splits_are_enforced(self):
        with self.assertRaises(ValueError):
            validate(Plan(source_height=2.5,target_heights=(2.5,3.1)))
        with self.assertRaises(ValueError):
            validate(Plan(training_forces=(1.0,1.6),target_forces=(1.0,2.0,3.0)))
        with self.assertRaises(ValueError):
            validate(Plan(training_seeds=(2026,2031),testing_seeds=(2031,2032)))
        with self.assertRaises(ValueError):
            validate(Plan(dt=.007))
        validate(Plan())

    def test_seed_streams_unique_for_train_pilot_and_eval(self):
        ids=[seed_for(seed,land,force,stream)
             for seed in (2026,2031,2032) for land in (0,1)
             for force in (0,1) for stream in (0,1,2)]
        self.assertEqual(len(ids),len(set(ids)))

    def test_exact_unbiased_reference_differs_with_height(self):
        vals=[exact_unbiased_mfpt(Config(height=h)) for h in (1.5,2.5,3.5)]
        self.assertTrue(np.isfinite(vals).all())
        self.assertTrue(vals[0] < vals[1] < vals[2])

    def test_effective_alpha_is_monotone(self):
        a=[effective_alpha(f, .25) for f in (0,1,2,3)]
        self.assertEqual(a[0],1.)
        self.assertTrue(a[0] > a[1] > a[2] > a[3])
        with self.assertRaises(ValueError):
            effective_alpha(2,-1)

    def test_kinetic_log_linear_extrapolation_control(self):
        alphas=np.array([.40,.65,.8])
        natural=8.5
        observed=np.exp(np.log(natural)+1.3*(alphas-1))
        self.assertAlmostEqual(kinetic_extrapolation(alphas,observed),natural,places=10)

    def test_censoring_causes_abstention(self):
        a=[.45,.65,.8]
        self.assertIsNone(kinetic_extrapolation(a,[2.,None,5.]))
        self.assertIsNone(kinetic_extrapolation(a,[2.,np.nan,5.]))
        self.assertIsNone(kinetic_extrapolation(a,[2.,0.,5.]))
        self.assertIsNone(kinetic_extrapolation([.45,.4501,.4502],[2.,3.,4.]))
        self.assertIsNone(kinetic_extrapolation(a[:2],[2.,5.]))

    def test_rank_correlation_is_descriptive(self):
        self.assertAlmostEqual(spearman([1,2,3,4],[2,4,6,8]),1.)
        self.assertAlmostEqual(spearman([1,2,3,4],[8,6,4,2]),-1.)
        self.assertIsNone(spearman([1,2],[4,5]))
        self.assertIsNone(spearman([1,1,1,1],[4,5,6,7]))
        np.testing.assert_allclose(rank([4,2,2,1]),[4,2.5,2.5,1])

    def test_config_time_alignment(self):
        p=Plan(dt=.01,horizon=40.,checkpoint_time=.1)
        c=cfg_for(p,3.1,1.9)
        self.assertAlmostEqual(c.dt*c.steps,40)
        self.assertAlmostEqual(c.dt*c.check_steps,.1)
        self.assertEqual(c.height,3.1)
        self.assertEqual(c.force,1.9)

    def test_small_end_to_end_never_refits_target_models(self):
        plan=Plan(replicas=12,horizon=8.,
                  training_seeds=(2026,2027),testing_seeds=(2031,),
                  target_heights=(1.7,3.1))
        orig_fit=RidgeLogistic.fit
        n_fit=[0]
        def count_fit(model,X,y):
            n_fit[0]+=1
            return orig_fit(model,X,y)
        with tempfile.TemporaryDirectory() as temp, patch.object(RidgeLogistic,"fit",count_fit):
            output=Path(temp)
            overview,events,seeds,landscapes=evaluate(plan,output)
            self.assertEqual(n_fit[0],2)  # once each on source; NEVER on targets
            self.assertFalse(overview["classifier_refitted_on_targets"])
            self.assertEqual(overview["source_height"],2.5)
            self.assertEqual(len(landscapes),2)
            self.assertEqual(len(events),2*3*1)
            self.assertEqual(len(seeds),2)
            self.assertTrue(all(row["height"]!=2.5 for row in events))
            self.assertTrue(all(row["n_checkpoint_events"]>0 for row in events))
            for file in ("frozen_source_models.npz","per_force_heldout.csv",
                         "per_seed_reconstruction.csv","landscape_summary.csv",
                         "overview.json","heldout_reliability.png","LIMITATIONS.txt"):
                self.assertTrue((output/file).is_file(),file)
            with (output/"overview.json").open() as fp:
                saved=json.load(fp)
            self.assertEqual(saved["n_training_checkpoint_events"],
                             overview["n_training_checkpoint_events"])
            with (output/"per_force_heldout.csv").open(newline="") as fp:
                self.assertEqual(len(list(csv.DictReader(fp))),6)
            with np.load(output/"frozen_source_models.npz") as saved:
                self.assertEqual(float(saved["training_height"]),2.5)


if __name__ == "__main__":
    unittest.main()
