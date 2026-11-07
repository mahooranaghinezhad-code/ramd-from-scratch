"""Numerical and scientific guards for held-out topology / path weighting."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from research import topology_girsanov_benchmark as m
except ImportError:
    import topology_girsanov_benchmark as m


class TopologyReweightingTests(unittest.TestCase):
    def test_independent_target_shapes_and_forces(self):
        m.validate(m.Design())
        for d in (
            replace(m.Design(), targets=('double_well','harmonic')),
            replace(m.Design(), train_forces=(.7,), test_forces=(.7,1.4,2.0)),
            replace(m.Design(), train_seeds=(2026,2031)),
            replace(m.Design(), dt=.007),
        ):
            with self.assertRaises(ValueError):
                m.validate(d)

    def test_analytic_gradient_of_each_distinct_topology(self):
        x=np.array([-1.45,-.8,-.32,.0,.35,.65])
        h=1e-5
        for name in ('harmonic','double_well','tilted_double','two_barriers','gaussian_gate'):
            with self.subTest(name=name):
                fd=(m.energy(x+h,name)-m.energy(x-h,name))/(2*h)
                np.testing.assert_allclose(m.grad_u(x,name),fd,atol=1e-7,rtol=1e-7)

    def test_exact_flat_reflecting_absorbing_mean(self):
        d=m.Design()
        expected=((d.right-d.start)*(d.right+d.start-2*d.left))/(2*d.kbt)
        self.assertAlmostEqual(m.full_mfpt('flat',d),expected,delta=.001)

    def test_restricted_pde_converges_and_below_unrestricted(self):
        d=m.Design(horizon=12.)
        for shape in ('harmonic','two_barriers','gaussian_gate'):
            with self.subTest(shape=shape):
                first=m.restricted_mfpt(shape,d,grid_n=201,time_steps=200)
                last=m.restricted_mfpt(shape,d,grid_n=401,time_steps=500)
                self.assertLess(abs(first-last),.035)
                self.assertLessEqual(last,m.full_mfpt(shape,d)+.008)
                self.assertGreater(last,0)

    def test_gaussian_ratio_is_exact_stepwise_density_ratio(self):
        force=1.3;dt=.01;kbt=1.4
        n=np.array([.6,.8]);z=np.array([1.05,-.3])
        sigma=np.sqrt(2*kbt*dt)
        raw_increment=force*n*dt+sigma*z
        logp=-np.dot(raw_increment,raw_increment)/(4*kbt*dt)
        logq=-np.dot(raw_increment-force*n*dt,raw_increment-force*n*dt)/(4*kbt*dt)
        got=m.gaussian_log_increment(force,kbt,dt,*n,*z)
        self.assertAlmostEqual(got,logp-logq,places=13)

    def test_simple_gaussian_importance_identity(self):
        rng=np.random.default_rng(18)
        N=150000
        z=rng.normal(size=(N,2))
        n=np.array([1.,0.]);F=.5;dt=.1;kbt=1.
        logw=m.gaussian_log_increment(F,kbt,dt,n[0],n[1],z[:,0],z[:,1])
        w=np.exp(logw)
        qx=F*dt+np.sqrt(2*kbt*dt)*z[:,0]
        self.assertAlmostEqual(w.mean(),1.,delta=.012)
        self.assertAlmostEqual(np.mean(w*qx),0.,delta=.014)

    def test_zero_force_weights_and_path_times(self):
        d=m.Design(replicas=40,horizon=2.)
        r=m.simulate('harmonic',d,force=0.,seed=2026,collect=True)
        self.assertTrue((r['logw']==0).all())
        self.assertTrue(np.isfinite(r['stop']).all())
        self.assertAlmostEqual(r['rmst'],np.mean(r['stop']))
        self.assertEqual(r['X'].shape[1],5)
        self.assertEqual(len(r['X']),len(r['y']))

    def test_path_weights_normalization_and_ess_limits(self):
        d=m.Design(replicas=160,horizon=3.)
        r=m.simulate('harmonic',d,force=.45,seed=9)
        self.assertTrue(np.isfinite(r['logw']).all())
        self.assertGreaterEqual(r['ess'],1.)
        self.assertLessEqual(r['ess'],d.replicas+1e-7)
        self.assertGreater(r['snis_rmst'],0.)
        self.assertLessEqual(r['snis_rmst'],d.horizon)
        self.assertGreater(r['max_weight_fraction'],0.)

    def test_direct_unbiased_simulation_agrees_with_pde_within_mc_error(self):
        d=m.Design(replicas=2600,horizon=5.)
        r=m.simulate('flat',d,force=0.,seed=444)
        expected=m.restricted_mfpt('flat',d,grid_n=351,time_steps=400)
        self.assertLess(abs(expected-r['rmst']),.22)

    def test_alpha_heuristic_abstains_or_recovers_exact_synthetic_fit(self):
        a=np.array([.55,.75,.88]); target=4.6
        synthetic=np.exp(np.log(target)+1.2*(a-1))
        self.assertAlmostEqual(m.extrapolate_alpha_rmstr(a,synthetic),target,places=10)
        self.assertIsNone(m.extrapolate_alpha_rmstr(a,[1.,np.nan,2.]))
        self.assertIsNone(m.extrapolate_alpha_rmstr(a[:2],[2.,3.]))
        self.assertIsNone(m.extrapolate_alpha_rmstr([.55,.551,.552],[1.,2.,3.]))

    def test_models_are_frozen_on_new_potential_topologies(self):
        d=m.Design(targets=('harmonic','two_barriers'),replicas=20,horizon=6.,
                   train_seeds=(2026,2027),test_seeds=(2031,2032))
        original=m.RidgeLogistic.fit
        count=[0]
        def counting(self,X,y):
            count[0]+=1
            return original(self,X,y)
        with tempfile.TemporaryDirectory() as temp,patch.object(m.RidgeLogistic,'fit',counting):
            p=Path(temp)
            info,ref,forces,seeds,shapes=m.evaluate(d,p)
            self.assertEqual(count[0],2)
            self.assertEqual(info['target_refit_count'],0)
            self.assertEqual(len(shapes),2)
            self.assertEqual(len(seeds),4)
            self.assertEqual(len(forces),12)
            self.assertTrue(all(s['shape']!='double_well' for s in seeds))
            self.assertTrue(all(np.isfinite(s['exact_rmst']) for s in seeds))
            for file in ('frozen_source_models.npz','exact_references.csv','per_force.csv',
                         'per_seed.csv','topology_summary.csv','overview.json',
                         'topology_and_reweighting.png','LIMITATIONS.txt'):
                self.assertTrue((p/file).is_file(),file)
            with (p/'overview.json').open() as stream:
                info2=json.load(stream)
            self.assertEqual(info2['source_fit_count'],2)
            self.assertTrue(info2['no_claim_of_new_method'])

if __name__ == '__main__':
    unittest.main()
