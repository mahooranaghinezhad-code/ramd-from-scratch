"""Scientific and software regression checks for coupled path reweighting.

Passing these tests validates implementation identities, not publication-grade
kinetic accuracy or an improvement of truncation for arbitrary weights.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from research import ess_scaling_coupled as m


class CoupledESSChecks(unittest.TestCase):
    def test_coupled_gradients_finite_difference(self):
        p=m.Plan()
        x=np.array([-.9,-.1,.3,.7])
        y=np.array([-.7,.4,-.2,.5])
        h=1e-6
        for coupling in (0.,.6,1.2):
            gx,gy=m.gradient(x,y,coupling,p)
            dx=(m.potential(x+h,y,coupling,p)-m.potential(x-h,y,coupling,p))/(2*h)
            dy=(m.potential(x,y+h,coupling,p)-m.potential(x,y-h,coupling,p))/(2*h)
            np.testing.assert_allclose(gx,dx,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(gy,dy,rtol=1e-7,atol=1e-7)

    def test_genuinely_nonseparable_potential(self):
        p=m.Plan()
        x,y=-.3,.6
        h=1e-5
        def mixed(c):
            return (m.potential(x+h,y+h,c,p)-m.potential(x+h,y-h,c,p)
                   -m.potential(x-h,y+h,c,p)+m.potential(x-h,y-h,c,p))/(4*h*h)
        self.assertAlmostEqual(mixed(0.),0.,delta=5e-6)
        self.assertAlmostEqual(mixed(1.2),-p.ky*1.2,delta=5e-6)

    def test_gaussian_density_ratio_anisotropic(self):
        f=1.4;dt=.01;kbt=1.;mu=.4
        n=np.array([.6,.8]);z=np.array([.7,-.3])
        mobility=np.array([1.,mu])
        sigma=np.sqrt(2*kbt*dt*mobility)
        raw=mobility*f*n*dt+sigma*z
        logp=-.5*np.sum((raw/sigma)**2)
        logq=-.5*np.sum(((raw-mobility*f*n*dt)/sigma)**2)
        predicted=m.log_gaussian_ratio(f,dt,kbt,mu,*n,*z)
        self.assertAlmostEqual(predicted,logp-logq,places=13)

    def test_gaussian_weight_is_normalized_one_step(self):
        rng=np.random.default_rng(20)
        n=160000
        z=rng.normal(size=(n,2))
        F=.7;dt=.1;mu=.4
        lw=m.log_gaussian_ratio(F,dt,1.,mu,1.,0.,z[:,0],z[:,1])
        w=np.exp(lw)
        xstep=F*dt+np.sqrt(2*dt)*z[:,0]
        self.assertAlmostEqual(float(w.mean()),1.,delta=.012)
        self.assertAlmostEqual(float(np.mean(w*xstep)),0.,delta=.013)

    def test_zero_force_exact_weight_invariance(self):
        p=m.Plan(couplings=(1.2,),forces=(0.,),horizons=(.5,1.),
                 replicas=42,reference_replicas=50)
        saved=m.simulate_snapshots(p,1.2,0.,2026)
        for h,obs in saved.items():
            np.testing.assert_array_equal(obs["logw"],np.zeros(42))
            summary=m.importance_estimates(obs["logw"],obs["stop"])
            expected=float(np.mean(obs["stop"]))
            self.assertAlmostEqual(summary["snis_rmst"],expected,places=12)
            self.assertAlmostEqual(summary["ionides_style_rmst"],expected,places=12)
            self.assertAlmostEqual(summary["raw_is_rmst"],expected,places=12)
            self.assertAlmostEqual(summary["ess_fraction"],1.,places=12)
            self.assertAlmostEqual(summary["clipped_ess_fraction"],1.,places=12)

    def test_truncation_predeclared_changes_large_weights(self):
        logw=np.array([0.,0.,0.,0.,np.log(1e8)])
        t=np.array([0.,0.,0.,0.,100.])
        v=m.importance_estimates(logw,t)
        self.assertGreater(v["capped_fraction"],0)
        self.assertGreater(v["clipped_ess"],v["ess"])
        self.assertLess(v["ionides_style_rmst"],v["snis_rmst"])
        self.assertTrue(np.isfinite(v["log_mean_weight"]))

    def test_path_terminates_once_and_snapshots_nested(self):
        p=m.Plan(couplings=(1.2,),forces=(1.2,),horizons=(1.,2.,4.),
                 replicas=72,reference_replicas=50)
        saved=m.simulate_snapshots(p,1.2,1.2,52)
        for h in p.horizons:
            self.assertEqual(len(saved[h]["stop"]),p.replicas)
            self.assertTrue((saved[h]["stop"]<=h).all())
            self.assertTrue(np.isfinite(saved[h]["logw"]).all())
            self.assertEqual(saved[h]["escaped"]+saved[h]["censored"],72)
        self.assertTrue((saved[1.]["stop"]<=saved[2.]["stop"]).all())
        self.assertTrue((saved[2.]["stop"]<=saved[4.]["stop"]).all())
        # For paths already escaped before horizon 1, logweights stop updating.
        ids=np.flatnonzero(saved[1.]["stop"]<1.)
        if len(ids):
            np.testing.assert_array_equal(saved[1.]["logw"][ids],saved[4.]["logw"][ids])

    def test_heldout_reference_seeds_are_independent(self):
        with self.assertRaises(ValueError):
            m.validate(m.Plan(test_seeds=(2026,2027),reference_seeds=(2027,50261)))
        with self.assertRaises(ValueError):
            m.validate(m.Plan(dt=.007))
        with self.assertRaises(ValueError):
            m.validate(m.Plan(mobility_y=0.))
        m.validate(m.Plan())

    def test_both_couplings_change_x_dynamics(self):
        p=m.Plan(horizons=(3.,),replicas=90)
        uncoupled=m.simulate_snapshots(p,0.,0.,seed=314)
        coupled=m.simulate_snapshots(p,1.2,0.,seed=314)
        # A direct x<-y coupling alters the FIRST PASSAGE dynamics,
        # not just a force-component or diagnostic label.
        self.assertFalse(np.array_equal(uncoupled[3.]["stop"],coupled[3.]["stop"]))

    def test_archived_pr_pilot_is_internally_consistent(self):
        from collections import defaultdict
        p=Path("research")
        with (p/"ess_coupled_conditions_2026.csv").open(newline="") as fh:
            conditions=list(csv.DictReader(fh))
        with (p/"ess_coupled_per_seed_2026.csv").open(newline="") as fh:
            samples=list(csv.DictReader(fh))
        with (p/"ess_coupled_F0_reference_2026.csv").open(newline="") as fh:
            refs=list(csv.DictReader(fh))
        self.assertEqual(len(conditions),24)
        self.assertEqual(len(samples),72)
        self.assertEqual(len(refs),6)
        grouped=defaultdict(list)
        for r in samples:
            key=(float(r["coupling"]),float(r["force"]),float(r["horizon"]))
            grouped[key].append(r)
        self.assertEqual(len(grouped),24)
        for r in conditions:
            key=(float(r["coupling"]),float(r["force"]),float(r["horizon"]))
            group=grouped[key]
            self.assertEqual(len(group),3)
            mu=sum(float(s["ess_fraction"]) for s in group)/3
            self.assertAlmostEqual(mu,float(r["mean_ess_fraction"]),places=11)
            self.assertEqual({int(s["seed"]) for s in group},{2026,2027,2028})
        self.assertTrue(all(int(r["n_reference"])==10500 for r in refs))

    def test_small_full_sweep_writes_every_output(self):
        p=m.Plan(couplings=(0.,1.2),forces=(0.,.9),
                 horizons=(.5,1.),test_seeds=(2026,2027),
                 reference_seeds=(50261,50262),replicas=24,
                 reference_replicas=150)
        with tempfile.TemporaryDirectory() as temp:
            reference,per,summary=m.run(p,Path(temp))
            self.assertEqual(len(reference),4)
            self.assertEqual(len(per),16)
            self.assertEqual(len(summary),8)
            self.assertTrue(all(np.isfinite(r["ess_fraction"]) for r in per))
            self.assertTrue(all(0<r["clipped_ess_fraction"]<=1.0000001 for r in per))
            for name in ("reference_F0.csv","per_seed.csv","conditions.csv",
                         "design.json","ess_scaling_coupled.png","LIMITATIONS.txt"):
                self.assertTrue((Path(temp)/name).is_file(),name)
            with (Path(temp)/"design.json").open() as f:
                settings=json.load(f)
            self.assertEqual(settings["n_path"],24)
            with (Path(temp)/"per_seed.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),16)


if __name__=="__main__":
    unittest.main()
