"""Regression and scientific-design checks for next-checkpoint switch modeling."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research.feedback_vs_exponential import Config,simulate
from research.state_conditioned_switching import (
    Study, make_config, record_checkpoints, features, RidgeLogistic,
    metrics, evaluate, seed_for, COLUMNS
)


class StateConditionedSwitchTests(unittest.TestCase):
    def test_instrumented_dynamics_identical_to_baseline(self):
        c=Config(replicas=40,steps=600,dt=.005,check_steps=20,force=3.,min_progress=.25)
        pilot=simulate(c,'feedback',4)
        p=pilot['n_resets']/pilot['checkpoint_opportunities']
        for name in ('feedback','checkpoint_random'):
            kwargs={'switch_probability':p} if name=='checkpoint_random' else {}
            reference=simulate(c,name,106,**kwargs)
            logged=record_checkpoints(c,name,106,**kwargs)
            np.testing.assert_array_equal(np.isnan(reference['times']),np.isnan(logged['escape_times']))
            np.testing.assert_allclose(reference['times'],logged['escape_times'],equal_nan=True,atol=0,rtol=0)
            self.assertEqual(reference['n_resets'],logged['n_changed'])
            self.assertEqual(reference['checkpoint_opportunities'],logged['opportunities'])

    def test_feedback_label_comes_from_future_interval(self):
        c=Config(replicas=32,steps=800,force=2.,min_progress=.42)
        result=record_checkpoints(c,'feedback',2026)
        data=result['raw']
        self.assertGreater(len(data),100)
        np.testing.assert_array_equal(data[:,8].astype(bool),data[:,9]<c.min_progress)
        first=data[data[:,2]==1]
        self.assertGreater(len(first),0)
        np.testing.assert_allclose(first[:,4],c.start)
        np.testing.assert_allclose(first[:,5],0.)
        np.testing.assert_allclose(np.hypot(first[:,6],first[:,7]),1.,atol=1e-12)
        np.testing.assert_allclose(first[:,3],0.)

    def test_future_displacement_and_labels_never_in_features(self):
        c=Config(replicas=32,steps=600)
        raw=record_checkpoints(c,'feedback',54)['raw']
        a=features(raw,c,'state');t=features(raw,c,'time')
        corrupt=raw.copy();corrupt[:,8]=1-corrupt[:,8];corrupt[:,9]+=1000
        np.testing.assert_array_equal(features(corrupt,c,'state'),a)
        np.testing.assert_array_equal(features(corrupt,c,'time'),t)
        self.assertEqual(a.shape[1],10)

    def test_random_control_not_exact_displacement_threshold(self):
        c=Config(replicas=48,steps=700,min_progress=.55)
        actual=record_checkpoints(c,'checkpoint_random',7,switch_probability=.4)['raw']
        self.assertGreater(len(actual),100)
        self.assertGreater(np.sum(actual[:,8]!=(actual[:,9]<c.min_progress)),20)

    def test_predictions_are_probabilities_with_finite_metrics(self):
        rng=np.random.default_rng(17)
        x=rng.normal(size=(400,3)); y=rng.binomial(1,.4,size=400)
        model=RidgeLogistic().fit(x,y)
        prediction=model.predict(x)
        self.assertTrue(((prediction>0)&(prediction<1)).all())
        self.assertTrue(np.isfinite(metrics(y,prediction)).all())

    def test_constant_features_do_not_break_logistic(self):
        X=np.zeros((80,3));y=np.array([0,1]*40)
        model=RidgeLogistic().fit(X,y)
        np.testing.assert_allclose(model.predict(X),0.5,atol=1e-6)

    def test_pilot_seed_streams_are_disjoint(self):
        seen=[seed_for(seed,cell,stream) for cell in range(2)
              for seed in (2026,2027,2030,2031) for stream in (0,1,2)]
        self.assertEqual(len(seen),len(set(seen)))

    def test_training_and_evaluation_batch_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                evaluate(Study(train_seeds=(2026,),test_seeds=(2026,),replicas=8,horizon=1),Path(d))

    def test_archived_heldout_scores_are_internally_consistent(self):
        path=Path("research/state_conditioned_switch_heldout_seeds_2026.csv")
        with path.open(newline="",encoding="utf-8") as fp:
            entries=list(csv.DictReader(fp))
        self.assertEqual(len(entries),16)
        groups={}
        for row in entries:
            key=(float(row["force"]),float(row["threshold"]),row["protocol"])
            groups.setdefault(key,set()).add(int(row["heldout_seed"]))
            self.assertAlmostEqual(float(row["improvement_state_vs_time"]),
                float(row["logloss_time"])-float(row["logloss_state"]),places=10)
            self.assertGreater(int(row["n_events"]),100)
        self.assertEqual(len(groups),4)
        self.assertTrue(all(seeds=={2031,2032,2033,2034} for seeds in groups.values()))

    def test_end_to_end_quick_separate_protocols(self):
        study=Study(conditions=((3.,.25),),replicas=32,horizon=3.,
                    train_seeds=(2026,2027),test_seeds=(2030,2031))
        with tempfile.TemporaryDirectory() as d:
            scores,summary,cal=evaluate(study,Path(d))
            self.assertEqual(len(scores),4)
            self.assertEqual(len(summary),2)
            self.assertTrue(cal)
            self.assertTrue(all(s['n_event']>0 for s in scores))
            self.assertTrue(all(np.isfinite(s['logloss_state']) for s in scores))
            for name in ('heldout_seed_metrics.csv','summary.csv','state_calibration.csv',
                         'heldout_model_comparison.png','LIMITATIONS.txt'):
                self.assertTrue((Path(d)/name).is_file(),name)
            with (Path(d)/'summary.csv').open(newline='') as h:
                self.assertEqual(len(list(csv.DictReader(h))),2)


if __name__=='__main__':unittest.main()
