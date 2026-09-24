#
#   This file is part of do-mpc
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.

"""Regression tests for examples/lstm_surrogate_model.

Covers the "trained ANN as the do-mpc plant model" route for a recurrent network:

* the hand-written LSTM cell translation matches ``torch.nn.LSTMCell`` exactly;
* unrolling that cell matches ``torch.nn.LSTM`` over a whole sequence, which is
  what proves the recurrence semantics (not just a single step) are right;
* ``model._rhs_fun`` -- the function MPC actually integrates -- reproduces
  PyTorch;
* the closed loop is compared against a golden result file, following the same
  convention as every other test in this directory.

Determinism
-----------
The network weights are generated from ``numpy.random.RandomState`` and copied
into the torch modules, rather than drawn from the torch RNG. The numpy stream is
stable across releases while torch's is not, which keeps the golden file valid
when torch is upgraded. No training happens here, so the test is fast.
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

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

EXAMPLE_DIR = '../examples/lstm_surrogate_model/'

# Small hidden size: the conversion is size-agnostic and this keeps the MPC tiny.
N_HID = 4
N_STEPS = 12
SEED = 20240921
TOL_TORCH = 1e-5      # float32 round-trip through torch -> CasADi float64
TOL_GOLDEN = 1e-8     # same tolerance as the rest of the testing suite


def dm(value):
    """CasADi DM -> numpy array (avoids the casadi>=3.8 numpy-legacy warning)."""
    return np.array(ca.DM(value).full())


@unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed (pip install "do-mpc[full]")')
class TestLSTMSurrogateModel(unittest.TestCase):

    def setUp(self):
        """Import the example modules. Reload in case another test imported a
        same-named module from a different example directory first."""
        default_path = copy.deepcopy(sys.path)
        sys.path.append(EXAMPLE_DIR)
        import main as example_main
        self.example = reload(example_main)
        sys.path = default_path

        self.rng = np.random.RandomState(SEED)
        self.net, self.stats = self._deterministic_surrogate()
        self.model, self.f_pre = self.example.build_lstm_model(
            self.net, self.stats, n_hid=N_HID)

    # ------------------------------------------------------------------ setup
    def _deterministic_surrogate(self):
        """Build a Surrogate with fixed weights drawn from the numpy RNG."""
        net = self.example.Surrogate(n_in=2, n_hid=N_HID, n_out=1)
        rng = self.rng

        def t(*shape):
            return torch.tensor(rng.randn(*shape).astype(np.float32))

        with torch.no_grad():
            net.cell.weight_ih.copy_(0.30 * t(4 * N_HID, 2))
            net.cell.weight_hh.copy_(0.30 * t(4 * N_HID, N_HID))
            net.cell.bias_ih.copy_(0.10 * t(4 * N_HID))
            net.cell.bias_hh.copy_(0.10 * t(4 * N_HID))
            net.head.weight.copy_(0.50 * t(1, N_HID))
            net.head.bias.copy_(0.10 * t(1))

        stats = {
            'in_mu': np.array([0.0, 0.0]),
            'in_sd': np.array([0.006, 0.06]),
            'out_mu': 0.0,
            'out_sd': 0.006,
        }
        return net, stats

    def _normalized_input(self, p, u):
        raw = np.vstack([p, u]).astype(np.float64)
        norm = (raw - self.stats['in_mu'].reshape(2, 1)) / self.stats['in_sd'].reshape(2, 1)
        # The torch modules hold float32 weights, so the input must be float32 too.
        # CasADi works in float64 and receives the same (rounded) values.
        return norm.astype(np.float32)

    def _torch_step(self, p, u, h, c):
        with torch.no_grad():
            h_t, c_t = self.net.cell(torch.tensor(self._normalized_input(p, u)).T,
                                     (torch.tensor(h).T, torch.tensor(c).T))
            p_t = self.net.head(h_t).numpy().ravel()[0] * self.stats['out_sd'] \
                + self.stats['out_mu']
        return p_t, h_t.numpy().T, c_t.numpy().T

    def _random_state(self):
        return (np.array([[self.rng.uniform(-0.01, 0.01)]]),
                np.array([[self.rng.uniform(-0.1, 0.1)]]),
                self.rng.randn(N_HID, 1).astype(np.float32),
                self.rng.randn(N_HID, 1).astype(np.float32))

    # ------------------------------------------------------------------ tests
    def test_model_dimensions(self):
        self.assertEqual(self.model.n_x, 1 + 2 * N_HID)
        self.assertEqual(self.model.n_u, 1)
        self.assertEqual(self.model.model_type, 'discrete')
        self.assertTrue(self.model.flags['setup'])

    def test_lstm_cell_expression_matches_torch(self):
        """The hand-written translation must reproduce torch.nn.LSTMCell."""
        worst = 0.0
        for _ in range(300):
            p, u, h, c = self._random_state()
            p_ref, h_ref, c_ref = self._torch_step(p, u, h, c)
            p_got, h_got, c_got = self.f_pre(p, h, c, u)
            worst = max(worst,
                        abs(float(np.array(p_got).ravel()[0]) - p_ref),
                        float(np.abs(dm(h_got) - h_ref).max()),
                        float(np.abs(dm(c_got) - c_ref).max()))
        self.assertLess(worst, TOL_TORCH, 'max deviation from torch was %.2e' % worst)
        print('  cell vs torch.nn.LSTMCell  : max diff = %.2e' % worst)

    def test_model_rhs_fun_matches_torch(self):
        """model._rhs_fun is what MPC integrates; it must equal PyTorch."""
        worst = 0.0
        for _ in range(200):
            p, u, h, c = self._random_state()
            p_ref, h_ref, c_ref = self._torch_step(p, u, h, c)
            rhs = dm(self.model._rhs_fun(
                np.vstack([p, h, c]), u,
                np.zeros((self.model.n_z, 1)), np.zeros((self.model.n_tvp, 1)),
                np.zeros((self.model.n_p, 1)), np.zeros((self.model.n_w, 1))))
            worst = max(worst, abs(rhs[0, 0] - p_ref),
                        float(np.abs(rhs[1:1 + N_HID] - h_ref).max()),
                        float(np.abs(rhs[1 + N_HID:] - c_ref).max()))
        self.assertLess(worst, TOL_TORCH, 'max deviation from torch was %.2e' % worst)
        print('  model._rhs_fun vs torch    : max diff = %.2e' % worst)

    def test_unrolled_cell_matches_torch_lstm(self):
        """Rolling the cell forward must equal torch.nn.LSTM over a sequence.

        This is the test that actually validates the recurrence: matching a
        single step does not prove the hidden state is threaded correctly.
        """
        lstm = torch.nn.LSTM(2, N_HID, batch_first=True)
        with torch.no_grad():
            lstm.weight_ih_l0.copy_(self.net.cell.weight_ih)
            lstm.weight_hh_l0.copy_(self.net.cell.weight_hh)
            lstm.bias_ih_l0.copy_(self.net.cell.bias_ih)
            lstm.bias_hh_l0.copy_(self.net.cell.bias_hh)

        seq = self.rng.randn(1, N_STEPS, 2).astype(np.float32)
        with torch.no_grad():
            y_ref, (h_ref, c_ref) = lstm(torch.tensor(seq))
        y_ref = y_ref.numpy()[0]
        h_ref = h_ref.numpy()[:, 0, :].T
        c_ref = c_ref.numpy()[:, 0, :].T

        h = np.zeros((N_HID, 1))
        c = np.zeros((N_HID, 1))
        y_got = []
        for t in range(N_STEPS):
            h_out, c_out = self.example.lstm_cell_casadi(
                ca.DM(seq[0, t].reshape(2, 1)), ca.DM(h), ca.DM(c), self.net.cell)
            h, c = dm(h_out), dm(c_out)
            y_got.append(h.ravel())
        y_got = np.array(y_got)

        worst = max(float(np.abs(y_got - y_ref).max()),
                    float(np.abs(h - h_ref).max()),
                    float(np.abs(c - c_ref).max()))
        self.assertLess(worst, TOL_TORCH, 'max deviation over %d steps was %.2e'
                        % (N_STEPS, worst))
        print('  %d-step unroll vs torch.nn.LSTM: max diff = %.2e' % (N_STEPS, worst))

    def test_mpc_problem_size(self):
        mpc = self.example.build_mpc(self.model, n_horizon=5)
        n_x, n_u = self.model.n_x, self.model.n_u
        expected = (5 + 1) * n_x + 5 * n_u
        self.assertEqual(mpc.opt_x_num.cat.shape[0], expected)
        print('  MPC opt_x = %d (expected %d), opt_p = %d'
              % (mpc.opt_x_num.cat.shape[0], expected, mpc.opt_p_num.cat.shape[0]))

    def run_closed_loop(self, n_horizon=5, n_steps=N_STEPS):
        """Run the LSTM-model MPC closed loop. Returns (mpc, simulator, estimator).

        Kept as a method (rather than inlined in the test) so that the golden
        result file can be regenerated with exactly the same computation.
        """
        mpc = self.example.build_mpc(self.model, n_horizon=n_horizon)
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.settings.t_step = self.example.DT
        simulator.setup()
        estimator = do_mpc.estimator.StateFeedback(self.model)

        x0 = np.zeros((self.model.n_x, 1))
        x0[0, 0] = 0.006
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()
        simulator.set_initial_guess()

        x = x0.copy()
        for _ in range(n_steps):
            u0 = mpc.make_step(x)
            x = estimator.make_step(simulator.make_step(u0))
        return mpc, simulator, estimator

    def test_closed_loop(self):
        """Golden-file regression, same convention as the other tests here."""
        mpc, simulator, estimator = self.run_closed_loop()

        # Store results (from a reference run):
        # do_mpc.data.save_results([mpc, simulator, estimator],
        #                          'results_lstm_surrogate', overwrite=True)

        golden = './results/results_lstm_surrogate.pkl'
        if not os.path.isfile(golden):
            self.skipTest('golden result file %s is missing; regenerate it with '
                          'save_results(..., overwrite=True)' % golden)
        ref = do_mpc.data.load_results(golden)

        msg = ('Variable {var} of {module} differs from the reference run: '
               '{check}. Max diff is {max_diff:.4E}.')
        for var in ['_x', '_u', '_time']:
            for name, obj in [('MPC', mpc), ('Simulator', simulator),
                              ('Estimator', estimator)]:
                max_diff = np.max(np.abs(obj.data.__dict__[var]
                                         - ref[name.lower()].__dict__[var]), initial=0)
                check = max_diff < TOL_GOLDEN
                self.assertTrue(check, msg.format(var=var, module=name,
                                                  check=check, max_diff=max_diff))

        # Physical sanity checks that do not depend on the golden file.
        # Data power indexing refers to the variable structure only; time is
        # always axis 0 of the returned array (see do_mpc.data.Data.__getitem__).
        p = np.asarray(mpc.data['_x', 'p']).ravel()
        u = np.asarray(mpc.data['_u', 'u']).ravel()
        self.assertTrue(np.all(np.abs(p) <= self.example.P_BOUND + 1e-9),
                        'position left its bounds')
        self.assertTrue(np.all(np.abs(u) <= self.example.U_BOUND + 1e-9),
                        'input left its bounds')
        # Note: the weights here are RANDOM (see _deterministic_surrogate), so the
        # loop is not expected to reach the setpoint. This test guards the
        # numerics and the constraint handling, not the control performance.
        # Control performance is demonstrated by examples/lstm_surrogate_model/main.py.
        print('  closed loop (random weights): final p = %+.6f, |u|max = %.5f (bound %.2f)'
              % (p[-1], np.abs(u).max(), self.example.U_BOUND))

        try:
            do_mpc.data.save_results([mpc, simulator], 'test_save_lstm', overwrite=True)
        except Exception:
            raise Exception('save_results failed')


if __name__ == '__main__':
    unittest.main()