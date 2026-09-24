#
#   This file is part of do-mpc
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.

"""Regression tests for do_mpc.differentiator.

The sensitivities are validated against **central finite differences of the
solution map**, which is a self-contained reference: no golden files, and no
dependence on library versions beyond IPOPT's own convergence.

Two levels are covered:

* ``NLPDifferentiator`` on a hand-written CasADi NLP (the constrained Rosenbrock
  from the class docstring), independent of every other do-mpc module;
* ``DoMPCDifferentiator`` on a real ``do_mpc.controller.MPC``, checking that
  ``du0/dx0`` matches a finite-difference of ``mpc.make_step``.
"""

import copy
import os
import sys
import unittest

import numpy as np

from importlib import reload

do_mpc_path = '../'
if do_mpc_path not in sys.path:
    sys.path.append(do_mpc_path)

import casadi as ca
import do_mpc
from do_mpc.differentiator import NLPDifferentiator, DoMPCDifferentiator
from do_mpc.differentiator.helper import (NLPDifferentiatorSettings,
                                          NLPDifferentiatorStatus)

# Finite-difference step and tolerance. The reference re-solves an NLP, so the
# FD estimate carries solver noise; the tolerances below are deliberately loose
# enough to absorb it while still catching a wrong sensitivity by orders of
# magnitude.
FD_EPS = 1e-5
FD_TOL = 5e-3


def dm(value):
    return np.array(ca.DM(value).full())


# --------------------------------------------------------------------- NLP fixture
def build_rosenbrock_nlp():
    """The constrained Rosenbrock NLP from the NLPDifferentiator docstring."""
    x = ca.SX.sym('x', 2)
    p = ca.SX.sym('p', 1)
    f = (1 - x[0])**2 + 0.2 * (x[1] - x[0]**2)**2
    cons_inner = (x[0] + 0.5)**2 + x[1]**2
    g = ca.vertcat(p**2 / 4 - cons_inner, cons_inner - p**2)
    nlp = {'x': x, 'p': p, 'f': f, 'g': g}
    nlp_bounds = {
        'lbx': np.array([0.0, -ca.inf]).reshape(-1, 1),
        'ubx': np.array([ca.inf, ca.inf]).reshape(-1, 1),
        'lbg': np.array([-ca.inf, -ca.inf]).reshape(-1, 1),
        'ubg': np.array([0.0, 0.0]).reshape(-1, 1),
    }
    return nlp, nlp_bounds, x, p


def make_solver(nlp, nlp_bounds):
    return ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
        'error_on_fail': False})


class TestNLPDifferentiator(unittest.TestCase):
    """The generic differentiator, exercised without any other do-mpc module."""

    def setUp(self):
        self.nlp, self.nlp_bounds, self.x_sym, self.p_sym = build_rosenbrock_nlp()
        self.solver = make_solver(self.nlp, self.nlp_bounds)
        self.p0 = np.array([[1.0]])
        self.x0_guess = np.array([[0.5], [0.25]])
        self.solution = self.solver(x0=self.x0_guess, p=self.p0, **self.nlp_bounds)

    def solve_at(self, p_value):
        """Re-solve from a FIXED initial guess so the map p -> x* is deterministic."""
        result = self.solver(x0=self.x0_guess, p=np.array([[p_value]]), **self.nlp_bounds)
        return dm(result['x']).ravel()

    def finite_difference_dxdp(self):
        plus = self.solve_at(self.p0[0, 0] + FD_EPS)
        minus = self.solve_at(self.p0[0, 0] - FD_EPS)
        return ((plus - minus) / (2 * FD_EPS)).reshape(-1, 1)

    def test_constructor_requires_dict_inputs(self):
        with self.assertRaises(Exception):
            NLPDifferentiator('not a dict', self.nlp_bounds)
        with self.assertRaises(Exception):
            NLPDifferentiator(self.nlp, 'not a dict')

    def test_constructor_requires_mandatory_nlp_keys(self):
        incomplete = {'x': self.x_sym, 'p': self.p_sym, 'f': self.nlp['f']}
        with self.assertRaises(Exception):
            NLPDifferentiator(incomplete, self.nlp_bounds)

    def test_constructor_requires_mandatory_bound_keys(self):
        incomplete = {'lbx': self.nlp_bounds['lbx'], 'ubx': self.nlp_bounds['ubx']}
        with self.assertRaises(Exception):
            NLPDifferentiator(self.nlp, incomplete)

    def test_settings_and_status_types(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        self.assertIsInstance(diff.settings, NLPDifferentiatorSettings)
        self.assertIsInstance(diff.status, NLPDifferentiatorStatus)
        self.assertEqual(diff.settings.lin_solver, 'casadi')
        self.assertTrue(diff.settings.check_LICQ)

    def test_sensitivity_matches_finite_differences(self):
        """The core correctness test: dx/dp against a FD of the solution map."""
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        dx_dp, dlam_dp = diff.differentiate(self.solution, self.p0)
        reference = self.finite_difference_dxdp()
        got = dm(dx_dp).reshape(-1, 1)
        error = np.abs(got - reference).max()
        self.assertLess(error, FD_TOL,
                        'dx/dp = %s but finite differences give %s (diff %.2e)'
                        % (got.ravel(), reference.ravel(), error))
        print('  NLPDifferentiator dx/dp = %s | FD = %s | diff = %.2e'
              % (got.ravel(), reference.ravel(), error))

    def test_status_fields_are_populated(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        diff.differentiate(self.solution, self.p0)
        status = diff.status
        self.assertIsNotNone(status.LICQ, 'check_LICQ defaults to True')
        self.assertIsNotNone(status.SC, 'check_SC defaults to True')
        self.assertIsNotNone(status.residuals, 'track_residuals defaults to True')
        self.assertTrue(status.lse_solved)
        self.assertTrue(status.sym_KKT)

    def test_disabling_checks_leaves_status_none(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        diff.settings.check_LICQ = False
        diff.settings.check_SC = False
        diff.settings.track_residuals = False
        diff.differentiate(self.solution, self.p0)
        self.assertIsNone(diff.status.LICQ)
        self.assertIsNone(diff.status.SC)
        self.assertIsNone(diff.status.residuals)

    def test_linear_solver_variants_agree(self):
        """'casadi', 'scipy' and 'lstsq' must all produce the same sensitivity."""
        reference = self.finite_difference_dxdp()
        for name in ('casadi', 'scipy', 'lstsq'):
            diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
            diff.settings.lin_solver = name
            dx_dp, _ = diff.differentiate(self.solution, self.p0)
            got = dm(dx_dp).reshape(-1, 1)
            self.assertLess(np.abs(got - reference).max(), FD_TOL,
                            'lin_solver=%r deviates from FD by %.2e'
                            % (name, np.abs(got - reference).max()))
            print('  lin_solver=%-7s diff vs FD = %.2e' % (name, np.abs(got - reference).max()))

    def test_lagrange_multiplier_sensitivity_shape(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        dx_dp, dlam_dp = diff.differentiate(self.solution, self.p0)
        self.assertEqual(tuple(dm(dx_dp).shape), (2, 1))       # n_x x n_p
        self.assertEqual(dm(dlam_dp).shape[1], 1)              # n_lam x n_p

    def test_differentiate_validates_nlp_sol(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        with self.assertRaises(ValueError):
            diff.differentiate('not a dict', self.p0)
        with self.assertRaises(ValueError) as ctx:
            diff.differentiate({'x': self.solution['x']}, self.p0)
        self.assertIn('lam_g', str(ctx.exception))

    def test_differentiate_validates_p_num(self):
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        with self.assertRaises(ValueError):
            diff.differentiate(self.solution, 'not numeric')
        with self.assertRaises(ValueError):
            diff.differentiate(self.solution, np.array([[1.0], [2.0]]))

    def test_p_num_accepts_scalar_and_ndarray(self):
        """float / int / np.ndarray are all coerced to DM."""
        diff = NLPDifferentiator(self.nlp, self.nlp_bounds)
        from_scalar = dm(diff.differentiate(self.solution, 1.0)[0])
        from_array = dm(diff.differentiate(self.solution, np.array([[1.0]]))[0])
        from_dm = dm(diff.differentiate(self.solution, ca.DM(1.0))[0])
        np.testing.assert_allclose(from_scalar, from_array, atol=1e-10)
        np.testing.assert_allclose(from_scalar, from_dm, atol=1e-10)


# --------------------------------------------------------------- MPC-level fixture
N_STATES = 3


def build_oscillating_masses_model():
    """A tiny discrete LTI plant, defined inline so the test is self-contained."""
    model = do_mpc.model.Model('discrete', 'SX')
    x = model.set_variable('_x', 'x', shape=(N_STATES, 1))
    u = model.set_variable('_u', 'u', shape=(1, 1))
    A = np.array([[0.8, 0.1, 0.0],
                  [-0.3, 0.9, 0.1],
                  [0.0, -0.2, 0.7]])
    B = np.array([[0.0], [0.1], [0.05]])
    model.set_rhs('x', A @ x + B @ u)
    model.setup()
    return model


def build_mpc(model):
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = 0.1
    mpc.settings.n_horizon = 6
    mpc.settings.supress_ipopt_output()
    mpc.set_objective(lterm=ca.sumsqr(model.x['x']), mterm=ca.sumsqr(model.x['x']))
    mpc.set_rterm(u=0.1)
    mpc.bounds['lower', '_x', 'x'] = np.array([[-2.0], [-2.0], [-2.0]])
    mpc.bounds['upper', '_x', 'x'] = np.array([[2.0], [2.0], [2.0]])
    mpc.bounds['lower', '_u', 'u'] = -1.0
    mpc.bounds['upper', '_u', 'u'] = 1.0
    mpc.setup()
    return mpc


@unittest.skipUnless(True, '')
class TestDoMPCDifferentiator(unittest.TestCase):
    """The do-mpc wrapper, validated against a finite difference of make_step."""

    def setUp(self):
        self.model = build_oscillating_masses_model()
        self.mpc = build_mpc(self.model)
        self.x0 = np.array([[0.6], [-0.3], [0.2]])

    def u0_of(self, x0):
        """A deterministic x0 -> u0 map.

        ``set_initial_guess`` is called before every solve so that the warm-start
        shift ``make_step`` applies at the end of the previous call cannot leak
        into the next one. Without this the FD reference would be inconsistent.
        """
        self.mpc.x0 = x0
        self.mpc.u0 = np.zeros((1, 1))
        self.mpc.set_initial_guess()
        return np.array(self.mpc.make_step(x0)).reshape(-1, 1)

    def test_requires_an_optimizer(self):
        with self.assertRaises(Exception):
            DoMPCDifferentiator(self.model)

    def test_differentiate_returns_sensitivities(self):
        diff = DoMPCDifferentiator(self.mpc)
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.x0)
        dx_dp, dlam_dp = diff.differentiate()
        self.assertEqual(dm(dx_dp).shape[1], dm(diff._get_p_num()).shape[0])
        self.assertIsInstance(diff.status, NLPDifferentiatorStatus)

    def test_sens_num_power_indexing(self):
        """The documented way to pull du0/dx0 out of the sensitivity structure."""
        from casadi.tools import indexf
        diff = DoMPCDifferentiator(self.mpc)
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.x0)
        diff.differentiate()
        du0_dx0 = diff.sens_num['dxdp', indexf['_u', 0, 0], indexf['_x0']]
        self.assertEqual(tuple(dm(du0_dx0).shape), (1, N_STATES))

    def test_du0_dx0_matches_finite_differences(self):
        """End-to-end: the analytic sensitivity vs a FD of the MPC itself."""
        from casadi.tools import indexf
        diff = DoMPCDifferentiator(self.mpc)
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.x0)
        diff.differentiate()
        analytic = dm(diff.sens_num['dxdp', indexf['_u', 0, 0], indexf['_x0']]).reshape(1, -1)

        eps = 1e-4
        fd = np.zeros((1, N_STATES))
        for i in range(N_STATES):
            xp = self.x0.copy(); xp[i, 0] += eps
            xm = self.x0.copy(); xm[i, 0] -= eps
            fd[0, i] = (self.u0_of(xp)[0, 0] - self.u0_of(xm)[0, 0]) / (2 * eps)

        error = np.abs(analytic - fd).max()
        self.assertLess(error, 5e-2,
                        'du0/dx0 = %s but FD gives %s (diff %.2e)'
                        % (analytic.ravel(), fd.ravel(), error))
        print('  DoMPCDifferentiator du0/dx0 = %s' % np.array2string(analytic.ravel(), precision=5))
        print('  finite difference  du0/dx0 = %s' % np.array2string(fd.ravel(), precision=5))
        print('  max abs diff = %.2e' % error)

    def test_settings_are_configurable(self):
        diff = DoMPCDifferentiator(self.mpc)
        diff.settings.check_LICQ = False
        diff.settings.lin_solver = 'scipy'
        self.assertFalse(diff.settings.check_LICQ)
        self.assertEqual(diff.settings.lin_solver, 'scipy')


if __name__ == '__main__':
    unittest.main()