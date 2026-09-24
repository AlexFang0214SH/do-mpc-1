#
#   This file is part of do-mpc
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.

"""Regression tests for do_mpc.sysid.ann_to_dompc_model.

Two groups:

* spec validation -- pure Python, no torch, no ONNX export. These check that a
  malformed wiring spec is rejected with a message that names the problem, and
  in particular that a **width mismatch** is caught at build time instead of
  surfacing later as ``Error in powerIndex slicing`` from casadi.tools.structure.
* end-to-end -- export real torch modules, build the model, verify that
  ``model._rhs_fun`` reproduces PyTorch, and run an MPC closed loop. Skipped when
  torch is unavailable.
"""

import copy
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from importlib import reload

do_mpc_path = '../'
if do_mpc_path not in sys.path:
    sys.path.append(do_mpc_path)

import casadi as ca
import do_mpc
from do_mpc.sysid import (ann_to_dompc_model, validate_wiring,
                        set_constant_parameters, set_constant_tvp)
from do_mpc.sysid._anntomodel import _template_index_form

try:
    import torch
    import onnx
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

OPSET = 17
TOL = 1e-5


def dm(value):
    return np.array(ca.DM(value).full())


# --------------------------------------------------------------- spec fixtures
X_SPEC = [('x', 2), ('h', 4)]
U_SPEC = [('u', 1)]
WIRING_OK = [('xu', [('x', 'x'), ('u', 'u')]), ('h', [('x', 'h')])]
OUTPUTS_OK = [(0, 'x'), (1, 'h')]


class TestSpecValidation(unittest.TestCase):
    """validate_wiring is pure Python, so these run without torch or onnx."""

    def test_accepts_a_consistent_spec(self):
        validate_wiring(X_SPEC, U_SPEC, WIRING_OK, OUTPUTS_OK)

    def test_duplicate_state_names(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2), ('x', 3)], U_SPEC, WIRING_OK, OUTPUTS_OK)
        self.assertIn('duplicate', str(ctx.exception))

    def test_empty_input_spec_is_allowed(self):
        """A state-only model with no inputs is legal."""
        validate_wiring([('x', 2)], [], [('x', [('x', 'x')])], [(0, 'x')])

    def test_wiring_referencing_an_undeclared_variable(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC,
                            [('xu', [('x', 'nope')]), ('h', [('x', 'h')])], OUTPUTS_OK)
        message = str(ctx.exception)
        self.assertIn('nope', message)
        self.assertIn('not declared', message)

    def test_wiring_with_an_unknown_kind(self):
        """An unrecognised kind gets its own message, not a generic 'not declared'."""
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC,
                            [('xu', [('w', 'x')]), ('h', [('x', 'h')])], OUTPUTS_OK)
        message = str(ctx.exception)
        self.assertIn('not one of', message)
        self.assertIn('tvp', message)

    def test_wiring_with_an_undeclared_variable(self):
        """A valid kind but a name that was never declared."""
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC,
                            [('xu', [('x', 'ghost')]), ('h', [('x', 'h')])], OUTPUTS_OK)
        message = str(ctx.exception)
        self.assertIn('ghost', message)
        self.assertIn('not declared', message)

    def test_wiring_entry_shape(self):
        with self.assertRaises(Exception):
            validate_wiring(X_SPEC, U_SPEC, [('xu',)], OUTPUTS_OK)
        with self.assertRaises(Exception):
            validate_wiring(X_SPEC, U_SPEC, [('xu', []), ('h', [('x', 'h')])], OUTPUTS_OK)

    def test_undriven_state_is_rejected(self):
        """Every state needs a rhs, otherwise Model.setup() would fail later."""
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC, WIRING_OK, [(0, 'x')])
        message = str(ctx.exception)
        self.assertIn("'h'", message)
        self.assertIn('never be updated', message)

    def test_state_driven_twice_is_rejected(self):
        # Every state is driven here, but 'x' twice -- so the duplicate check
        # fires rather than the "never updated" one.
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], U_SPEC,
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x'), (1, 'x')])
        self.assertIn('twice', str(ctx.exception))

    def test_output_driving_a_non_state(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC, WIRING_OK, [(0, 'x'), (1, 'u')])
        self.assertIn('not a state', str(ctx.exception))

    def test_duplicate_onnx_input_in_wiring(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring(X_SPEC, U_SPEC,
                            [('xu', [('x', 'x'), ('u', 'u')]), ('xu', [('x', 'h')])],
                            OUTPUTS_OK)
        self.assertIn('twice', str(ctx.exception))

    def test_negative_size(self):
        with self.assertRaises(Exception):
            validate_wiring([('x', -1)], U_SPEC, WIRING_OK, OUTPUTS_OK)

    # -- parameters and time-varying parameters ------------------------------
    def test_wiring_accepts_p_and_tvp_kinds(self):
        validate_wiring([('x', 2)], [('u', 1)],
                        [('z', [('x', 'x'), ('u', 'u'), ('p', 'theta'), ('tvp', 'ref')])],
                        [(0, 'x')],
                        p_spec=[('theta', 2)], tvp_spec=[('ref', 1)])

    def test_p_referenced_without_p_spec_is_rejected(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('z', [('x', 'x'), ('u', 'u'), ('p', 'theta')])],
                            [(0, 'x')], p_spec=[])
        message = str(ctx.exception)
        self.assertIn('theta', message)
        self.assertIn('p_spec', message)

    def test_tvp_referenced_without_tvp_spec_is_rejected(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('z', [('x', 'x'), ('u', 'u'), ('tvp', 'ref')])],
                            [(0, 'x')], tvp_spec=[])
        self.assertIn('tvp_spec', str(ctx.exception))

    def test_z_kind_is_allowed_when_declared(self):
        """An implicit DAE 0 = g(x, u, z) must be able to read z."""
        validate_wiring([('x', 2)], [('u', 1)],
                        [('xu', [('x', 'x'), ('u', 'u'), ('z', 'alg')])], [(0, 'x')],
                        z_spec=[('alg', 2)], algebraic=[(1, 'alg')])

    def test_algebraic_state_without_an_equation_is_rejected(self):
        """do-mpc asserts n_z == number of algebraic equations, so catch it early."""
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                            z_spec=[('alg', 2)], algebraic=[])
        message = str(ctx.exception)
        self.assertIn('alg', message)
        self.assertIn('exactly one equation', message)

    def test_algebraic_target_must_be_declared(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                            z_spec=[('alg', 2)], algebraic=[(1, 'ghost')])
        self.assertIn('z_spec', str(ctx.exception))

    def test_algebraic_state_with_two_equations_is_rejected(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                            z_spec=[('alg', 2)], algebraic=[(1, 'alg'), (2, 'alg')])
        self.assertIn('more than one equation', str(ctx.exception))

    def test_measurement_spec_validation(self):
        validate_wiring([('x', 2)], [('u', 1)],
                        [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                        measurements=[(0, 'y1'), ((0, slice(0, 1)), 'y2', False)])
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                            measurements=[(0, 'y1'), (1, 'y1')])
        self.assertIn('twice', str(ctx.exception))
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')],
                            measurements=[(0, 'y1', 'not-a-bool')])
        self.assertIn('meas_noise must be a bool', str(ctx.exception))

    def test_z_kind_without_z_spec_is_rejected(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)],
                            [('xu', [('z', 'alg')])], [(0, 'x')], z_spec=[])
        self.assertIn('z_spec', str(ctx.exception))

    def test_noise_kinds_are_rejected_with_an_explanation(self):
        """_w / _v are generated by do-mpc and cannot be network inputs."""
        for kind in ('w', 'v'):
            with self.assertRaises(Exception) as ctx:
                validate_wiring([('x', 2)], [('u', 1)],
                                [('xu', [(kind, 'x')])], [(0, 'x')])
            self.assertIn('Noise variables', str(ctx.exception),
                          'kind %r produced an unclear error' % kind)

    def test_duplicate_parameter_names(self):
        with self.assertRaises(Exception) as ctx:
            validate_wiring([('x', 2)], [('u', 1)], WIRING_OK, OUTPUTS_OK,
                            p_spec=[('theta', 1), ('theta', 2)])
        self.assertIn('duplicate', str(ctx.exception))


@unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed (pip install "do-mpc[full]")')
class TestAnnToDompcModel(unittest.TestCase):
    """End-to-end: torch module -> ONNX -> do_mpc Model -> MPC closed loop."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.rng = np.random.RandomState(4242)

    # ------------------------------------------------------------ plant models
    @staticmethod
    def ffn(n_in=3, n_out=2, units=16):
        return torch.nn.Sequential(torch.nn.Linear(n_in, units), torch.nn.Tanh(),
                                   torch.nn.Linear(units, units), torch.nn.Tanh(),
                                   torch.nn.Linear(units, n_out))

    @staticmethod
    def residual(n_in=3, n_out=2, units=8):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project_in = torch.nn.Linear(n_in, units)
                self.linear = torch.nn.Linear(units, units)
                self.activation = torch.nn.Tanh()
                self.project_out = torch.nn.Linear(units, n_out)

            def forward(self, z):
                hidden = self.project_in(z)
                return self.project_out(hidden + self.activation(self.linear(hidden)))
        return Residual()

    @staticmethod
    def lstm_plant(n_obs=2, n_u=1, n_hid=4):
        class LSTMPlant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = torch.nn.LSTMCell(n_obs + n_u, n_hid)
                self.head = torch.nn.Linear(n_hid, n_obs)

            def forward(self, xu, h, c):
                h_next, c_next = self.cell(xu, (h, c))
                return self.head(h_next), h_next, c_next
        return LSTMPlant()

    @staticmethod
    def gru_plant(n_obs=2, n_u=1, n_hid=4):
        class GRUPlant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cell = torch.nn.GRUCell(n_obs + n_u, n_hid)
                self.head = torch.nn.Linear(n_hid, n_obs)

            def forward(self, xu, h):
                h_next = self.cell(xu, h)
                return self.head(h_next), h_next
        return GRUPlant()

    # ------------------------------------------------------------------ helpers
    def rhs_of(self, model, x, u):
        return dm(model._rhs_fun(x, u, np.zeros((model.n_z, 1)), np.zeros((model.n_tvp, 1)),
                                 np.zeros((model.n_p, 1)), np.zeros((model.n_w, 1))))

    def check_against_torch(self, module, model, x_parts, u_size, n_points=150):
        """model._rhs_fun must reproduce the torch module bit-for-bit (to float32)."""
        worst = 0.0
        for _ in range(n_points):
            parts = [self.rng.randn(*shape).astype(np.float32) for shape in x_parts]
            u = self.rng.randn(u_size, 1).astype(np.float32) if u_size else np.zeros((0, 1))
            x = np.vstack(parts) if len(parts) > 1 else parts[0]
            feed = [np.vstack([parts[0], u]).astype(np.float32)] + parts[1:]
            with torch.no_grad():
                # torch modules take row vectors (1, n); do-mpc uses columns (n, 1)
                reference = module(*[torch.tensor(f.T) for f in feed])
            reference = reference if isinstance(reference, tuple) else (reference,)
            got = self.rhs_of(model, x, u)
            offset = 0
            for part, ref in zip(x_parts, reference):
                width = part[0]
                array = ref.numpy()
                worst = max(worst, float(np.abs(got[offset:offset + width].reshape(array.shape)
                                                - array).max()))
                offset += width
        return worst

    # -------------------------------------------------------------------- tests
    def test_feed_forward_network(self):
        module = self.ffn()
        model, converter = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, 3),), input_names=['xu'])
        self.assertEqual(model.n_x, 2)
        self.assertEqual(model.n_u, 1)
        self.assertEqual(model.model_type, 'discrete')
        self.assertTrue(model.flags['setup'])
        worst = self.check_against_torch(module, model, [(2, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  FFN          n_x=%d  max|rhs_fun - torch| = %.2e' % (model.n_x, worst))

    def test_residual_network(self):
        module = self.residual()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, 3),), input_names=['xu'])
        worst = self.check_against_torch(module, model, [(2, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  Residual     n_x=%d  max|rhs_fun - torch| = %.2e' % (model.n_x, worst))

    def test_lstm_hidden_state_becomes_a_model_state(self):
        module = self.lstm_plant()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2), ('h', 4), ('c', 4)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')]), ('h', [('x', 'h')]),
                    ('c', [('x', 'c')])],
            outputs=[(0, 'x'), (1, 'h'), (2, 'c')],
            sample_args=(torch.zeros(1, 3), torch.zeros(1, 4), torch.zeros(1, 4)),
            input_names=['xu', 'h', 'c'])
        self.assertEqual(model.n_x, 2 + 4 + 4)
        worst = self.check_against_torch(module, model, [(2, 1), (4, 1), (4, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  LSTMCell     n_x=%d  max|rhs_fun - torch| = %.2e' % (model.n_x, worst))

    def test_gru_hidden_state_becomes_a_model_state(self):
        module = self.gru_plant()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2), ('h', 4)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')]), ('h', [('x', 'h')])],
            outputs=[(0, 'x'), (1, 'h')],
            sample_args=(torch.zeros(1, 3), torch.zeros(1, 4)),
            input_names=['xu', 'h'])
        self.assertEqual(model.n_x, 2 + 4)
        worst = self.check_against_torch(module, model, [(2, 1), (4, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  GRUCell      n_x=%d  max|rhs_fun - torch| = %.2e' % (model.n_x, worst))

    def test_accepts_an_onnx_file_path(self):
        module = self.ffn()
        tmp = tempfile.mkdtemp(prefix='_onnx_tmp_', dir=os.getcwd())
        try:
            path = os.path.join(tmp, 'ffn.onnx')
            module.eval()
            torch.onnx.export(module, (torch.zeros(1, 3),), path, input_names=['xu'],
                              dynamo=False, opset_version=OPSET)
            model, _ = ann_to_dompc_model(
                path, x_spec=[('x', 2)], u_spec=[('u', 1)],
                wiring=[('xu', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')])
            worst = self.check_against_torch(module, model, [(2, 1)], 1)
            self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_accepts_a_preloaded_model_proto(self):
        module = self.ffn()
        module.eval()
        proto = do_mpc.sysid.torch_module_to_onnx(module, (torch.zeros(1, 3),), ['xu'])
        model, _ = ann_to_dompc_model(
            proto, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')])
        worst = self.check_against_torch(module, model, [(2, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)

    def test_mx_symvar_type(self):
        """symvar_type='MX' must work, not just be accepted as an argument."""
        module = self.ffn()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, 3),), input_names=['xu'], symvar_type='MX')
        self.assertEqual(model.symvar_type, 'MX')
        worst = self.check_against_torch(module, model, [(2, 1)], 1)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  MX symvar    n_x=%d  max|rhs_fun - torch| = %.2e' % (model.n_x, worst))

    def test_state_only_model_without_inputs(self):
        """An autonomous surrogate (no _u) must build and simulate."""
        module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 4))
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 4)], u_spec=[],
            wiring=[('xu', [('x', 'x')])], outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, 4),), input_names=['xu'])
        self.assertEqual(model.n_u, 0)
        worst = self.check_against_torch(module, model, [(4, 1)], 0)
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)

    def test_continuous_model_type_warns(self):
        """The discrete/continuous choice is a semantic claim the builder cannot
        verify, so it must at least warn."""
        module = self.ffn()
        with self.assertWarns(UserWarning) as ctx:
            ann_to_dompc_model(module, x_spec=[('x', 2)], u_spec=[('u', 1)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')], model_type='continuous',
                               sample_args=(torch.zeros(1, 3),), input_names=['xu'])
        message = str(ctx.warning)
        self.assertIn('dx/dt', message)
        self.assertIn("model_type='discrete'", message)

    def test_discrete_model_type_does_not_warn(self):
        module = self.ffn()
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter('error', UserWarning)
            ann_to_dompc_model(module, x_spec=[('x', 2)], u_spec=[('u', 1)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')],
                               sample_args=(torch.zeros(1, 3),), input_names=['xu'])

    def test_noise_kind_is_rejected_with_an_explanation(self):
        """_w / _v are generated by do-mpc; wiring one must say so clearly."""
        module = self.ffn()
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(module, x_spec=[('x', 2)], u_spec=[('u', 1)],
                               wiring=[('xu', [('w', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')],
                               sample_args=(torch.zeros(1, 3),), input_names=['xu'])
        self.assertIn('Noise variables', str(ctx.exception))

    def test_p_and_tvp_kinds_are_now_supported(self):
        """Regression guard: these used to be rejected as undeclared."""
        module = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 2))
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            p_spec=[('theta', 2)], tvp_spec=[('ref', 1)],
            wiring=[('xu', [('x', 'x'), ('u', 'u'), ('p', 'theta'), ('tvp', 'ref')])],
            outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, 6),), input_names=['xu'])
        self.assertEqual(model.n_p, 2)
        self.assertEqual(model.n_tvp, 1)

    # ------------------------------------------------------- error diagnostics
    def test_width_mismatch_is_caught_at_build_time(self):
        """The message that replaces an opaque casadi powerIndex error.

        The network takes 6 (= 5 states + 1 input, matching the wiring) and
        returns 2, but the spec declares a 5-wide state. Only the OUTPUT width
        may disagree here, otherwise the wiring check fires first.
        """
        module = self.ffn(n_in=6, n_out=2)
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(module, x_spec=[('x', 5)], u_spec=[('u', 1)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')],
                               sample_args=(torch.zeros(1, 6),), input_names=['xu'])
        message = str(ctx.exception)
        self.assertIn('wiring mismatch', message)
        self.assertIn('is 2-wide', message)
        self.assertIn('Graph outputs are', message)

    def test_wiring_width_mismatch_is_caught(self):
        """The graph input is 4-wide but the spec concatenates 2 + 3 = 5."""
        module = self.ffn(n_in=4, n_out=2)
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(module, x_spec=[('x', 2)], u_spec=[('u', 3)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')],
                               sample_args=(torch.zeros(1, 4),), input_names=['xu'])
        message = str(ctx.exception)
        self.assertIn('concatenation of', message)
        self.assertIn('4-wide', message)

    def test_unwired_graph_input_is_caught(self):
        module = self.lstm_plant()
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(module, x_spec=[('x', 2), ('h', 4), ('c', 4)],
                               u_spec=[('u', 1)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')]),
                                       ('h', [('x', 'h')])],       # 'c' missing
                               outputs=[(0, 'x'), (1, 'h'), (2, 'c')],
                               sample_args=(torch.zeros(1, 3), torch.zeros(1, 4),
                                            torch.zeros(1, 4)),
                               input_names=['xu', 'h', 'c'])
        self.assertIn('must be wired exactly once', str(ctx.exception))

    def test_missing_sample_args_for_torch_module(self):
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(self.ffn(), x_spec=[('x', 2)], u_spec=[('u', 1)],
                               wiring=[('xu', [('x', 'x'), ('u', 'u')])],
                               outputs=[(0, 'x')])
        self.assertIn('sample_args', str(ctx.exception))

    def test_bad_source_type(self):
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(42, x_spec=[('x', 2)], u_spec=[], wiring=[], outputs=[])
        self.assertIn('source must be', str(ctx.exception))

    # ------------------------------------------------------------ MPC closed loop
    def _closed_loop(self, model, n_steps=8, n_horizon=6):
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = n_horizon
        mpc.settings.supress_ipopt_output()
        mpc.set_objective(lterm=ca.sumsqr(model.x['x']), mterm=ca.sumsqr(model.x['x']))
        if model.n_u:
            mpc.set_rterm(**{name: 0.05 for name, _ in
                             [(n, 0) for n in model.u.keys() if n != 'default']})
        for name in [n for n, _ in [(n, 0) for n in model.x.keys() if n != 'default']]:
            size = model.x[name].shape[0]
            bound = 5.0 if name == 'x' else 50.0
            mpc.bounds['lower', '_x', name] = -bound * np.ones((size, 1))
            mpc.bounds['upper', '_x', name] = bound * np.ones((size, 1))
        for name in [n for n in model.u.keys() if n != 'default']:
            mpc.bounds['lower', '_u', name] = -2.0
            mpc.bounds['upper', '_u', name] = 2.0
        mpc.setup()

        simulator = do_mpc.simulator.Simulator(model)
        simulator.settings.t_step = 0.1
        simulator.setup()
        estimator = do_mpc.estimator.StateFeedback(model)

        x0 = np.zeros((model.n_x, 1))
        x0[0, 0] = 0.7
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()
        simulator.set_initial_guess()

        x = x0.copy()
        inputs = []
        for _ in range(n_steps):
            u0 = mpc.make_step(x)
            inputs.append(np.array(u0).ravel())
            x = estimator.make_step(simulator.make_step(u0))
        return mpc, x, np.array(inputs)

    def test_mpc_closed_loop_on_each_architecture(self):
        cases = [
            ('FFN', self.ffn(), [('x', 2)], [('u', 1)],
             [('xu', [('x', 'x'), ('u', 'u')])], [(0, 'x')], (torch.zeros(1, 3),), ['xu']),
            ('LSTM', self.lstm_plant(), [('x', 2), ('h', 4), ('c', 4)], [('u', 1)],
             [('xu', [('x', 'x'), ('u', 'u')]), ('h', [('x', 'h')]), ('c', [('x', 'c')])],
             [(0, 'x'), (1, 'h'), (2, 'c')],
             (torch.zeros(1, 3), torch.zeros(1, 4), torch.zeros(1, 4)), ['xu', 'h', 'c']),
            ('GRU', self.gru_plant(), [('x', 2), ('h', 4)], [('u', 1)],
             [('xu', [('x', 'x'), ('u', 'u')]), ('h', [('x', 'h')])],
             [(0, 'x'), (1, 'h')],
             (torch.zeros(1, 3), torch.zeros(1, 4)), ['xu', 'h']),
        ]
        for name, module, x_spec, u_spec, wiring, outputs, sample_args, names in cases:
            model, _ = ann_to_dompc_model(module, x_spec, u_spec, wiring, outputs,
                                          sample_args=sample_args, input_names=names)
            mpc, x_final, inputs = self._closed_loop(model)
            self.assertTrue(np.all(np.isfinite(x_final)), '%s produced non-finite states' % name)
            self.assertTrue(np.all(np.abs(inputs) <= 2.0 + 1e-6), '%s violated input bounds' % name)
            # the objective is sum(x^2) minimisation from x0 = 0.7, so the state
            # must move towards zero
            self.assertLess(abs(x_final[0, 0]), 0.7, '%s did not move towards the origin' % name)
            print('  MPC closed loop %-5s n_x=%2d opt_x=%4d  x0=0.70000 -> xN=%+.5f'
                  % (name, model.n_x, mpc.opt_x_num.cat.shape[0], x_final[0, 0]))


@unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed (pip install "do-mpc[full]")')
class TestParameterWiring(unittest.TestCase):
    """Networks conditioned on ``_p`` and ``_tvp``.

    ``_p`` is constant over the whole horizon and can be an uncertainty for
    robust multi-stage MPC; ``_tvp`` takes a different value at every horizon
    step and is how a reference trajectory or a forecast enters the model.
    """

    N_X, N_U, N_P, N_TVP = 2, 1, 2, 1
    NET_IN = N_X + N_U + N_P + N_TVP

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.rng = np.random.RandomState(1717)
        self.module = self.conditioned_net(self.NET_IN, self.N_X)
        self.model, _ = ann_to_dompc_model(
            self.module,
            x_spec=[('x', self.N_X)], u_spec=[('u', self.N_U)],
            p_spec=[('theta', self.N_P)], tvp_spec=[('ref', self.N_TVP)],
            wiring=[('z', [('x', 'x'), ('u', 'u'), ('p', 'theta'), ('tvp', 'ref')])],
            outputs=[(0, 'x')],
            sample_args=(torch.zeros(1, self.NET_IN),), input_names=['z'])

    @staticmethod
    def conditioned_net(n_in, n_out):
        module = torch.nn.Sequential(torch.nn.Linear(n_in, 12), torch.nn.Tanh(),
                                     torch.nn.Linear(12, n_out))
        module.eval()
        return module

    def rhs(self, model, x, u, p, tvp):
        return dm(model._rhs_fun(x, u, np.zeros((model.n_z, 1)), tvp, p,
                                 np.zeros((model.n_w, 1))))

    # ------------------------------------------------------------------ model
    def test_model_declares_p_and_tvp(self):
        self.assertEqual(self.model.n_x, self.N_X)
        self.assertEqual(self.model.n_u, self.N_U)
        self.assertEqual(self.model.n_p, self.N_P)
        self.assertEqual(self.model.n_tvp, self.N_TVP)

    def test_rhs_matches_torch_with_p_and_tvp(self):
        worst = 0.0
        for _ in range(200):
            x = self.rng.randn(self.N_X, 1).astype(np.float32)
            u = self.rng.randn(self.N_U, 1).astype(np.float32)
            p = self.rng.randn(self.N_P, 1).astype(np.float32)
            tvp = self.rng.randn(self.N_TVP, 1).astype(np.float32)
            z = np.vstack([x, u, p, tvp]).astype(np.float32)
            with torch.no_grad():
                reference = self.module(torch.tensor(z).T).numpy().ravel()
            worst = max(worst, float(np.abs(self.rhs(self.model, x, u, p, tvp).ravel()
                                            - reference).max()))
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  conditioned net: max|rhs_fun - torch| = %.2e' % worst)

    # ------------------------------------------------------- template asymmetry
    def _configured_mpc(self, n_robust=0, uncertainty=None, n_horizon=5):
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = n_horizon
        mpc.settings.supress_ipopt_output()
        if n_robust:
            mpc.settings.n_robust = n_robust
        mpc.set_objective(lterm=ca.sumsqr(self.model.x['x']),
                          mterm=ca.sumsqr(self.model.x['x']))
        mpc.set_rterm(u=0.01)
        mpc.bounds['lower', '_x', 'x'] = -5 * np.ones((self.N_X, 1))
        mpc.bounds['upper', '_x', 'x'] = 5 * np.ones((self.N_X, 1))
        mpc.bounds['lower', '_u', 'u'] = -2.0
        mpc.bounds['upper', '_u', 'u'] = 2.0
        if uncertainty is not None:
            mpc.set_uncertainty_values(theta=uncertainty)
        else:
            set_constant_parameters(mpc, theta=[0.4, -0.2])
        set_constant_tvp(mpc, ref=0.0)
        mpc.setup()
        return mpc

    def _configured_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.settings.t_step = 0.1
        # must be registered BEFORE setup()
        set_constant_parameters(simulator, theta=[0.4, -0.2])
        set_constant_tvp(simulator, ref=0.0)
        simulator.setup()
        return simulator

    def test_template_index_form_is_detected_per_class(self):
        """MPC templates have an extra leading dimension; Simulator ones do not."""
        mpc = self._configured_mpc()
        simulator = self._configured_simulator()
        self.assertEqual(_template_index_form(mpc.get_tvp_template(), '_tvp', 'ref'),
                         ('_tvp', slice(None), 'ref'))
        self.assertEqual(_template_index_form(mpc.get_p_template(1), '_p', 'theta'),
                         ('_p', slice(None), 'theta'))
        self.assertEqual(_template_index_form(simulator.get_tvp_template(), '_tvp', 'ref'),
                         ('ref', slice(None)))
        self.assertEqual(_template_index_form(simulator.get_p_template(), '_p', 'theta'),
                         ('theta', slice(None)))

    def test_set_constant_parameters_fills_every_scenario(self):
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = 3
        template = set_constant_parameters(mpc, n_combinations=3, theta=[1.0, 2.0])
        values = np.array(ca.DM(template).full()).ravel()
        self.assertEqual(values.size, 3 * self.N_P)
        np.testing.assert_allclose(values, [1.0, 2.0] * 3)

    def test_set_constant_tvp_fills_the_whole_horizon(self):
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = 4
        mpc.settings.supress_ipopt_output()
        template = set_constant_tvp(mpc, ref=0.75)
        values = np.array(ca.DM(template).full()).ravel()
        self.assertEqual(values.size, mpc.settings.n_horizon + 1)
        np.testing.assert_allclose(values, np.full(5, 0.75))

    def test_helpers_work_on_the_simulator_too(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.settings.t_step = 0.1
        p_template = set_constant_parameters(simulator, theta=[1.0, 2.0])
        tvp_template = set_constant_tvp(simulator, ref=0.5)
        np.testing.assert_allclose(np.array(ca.DM(p_template).full()).ravel(), [1.0, 2.0])
        np.testing.assert_allclose(np.array(ca.DM(tvp_template).full()).ravel(), [0.5])

    def test_tvp_can_be_changed_at_runtime(self):
        """A moving setpoint: reassign the template between make_step calls."""
        mpc = self._configured_mpc()
        simulator = self._configured_simulator()
        estimator = do_mpc.estimator.StateFeedback(self.model)
        x0 = np.zeros((self.model.n_x, 1))
        x0[0, 0] = 0.3
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()
        simulator.set_initial_guess()

        mpc_tvp = mpc.get_tvp_template()
        sim_tvp = simulator.get_tvp_template()
        mpc_index = _template_index_form(mpc_tvp, '_tvp', 'ref')
        sim_index = _template_index_form(sim_tvp, '_tvp', 'ref')

        x = x0.copy()
        seen = []
        for reference in (0.0, 0.05, -0.05):
            mpc_tvp[mpc_index] = reference
            sim_tvp[sim_index] = reference
            for _ in range(4):
                u0 = mpc.make_step(x)
                x = estimator.make_step(simulator.make_step(u0))
            seen.append(float(x[0, 0]))
            self.assertTrue(np.all(np.isfinite(x)), 'non-finite state at ref=%s' % reference)
        # the three setpoints must not all produce the identical state
        self.assertGreater(max(seen) - min(seen), 0.0,
                           'changing the tvp had no effect on the closed loop')
        print('  moving setpoint: x[0] tracked %s' % np.round(seen, 6))

    def test_mpc_closed_loop_with_parameters(self):
        mpc = self._configured_mpc()
        simulator = self._configured_simulator()
        estimator = do_mpc.estimator.StateFeedback(self.model)
        x0 = np.zeros((self.model.n_x, 1))
        x0[0, 0] = 0.3
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()
        simulator.set_initial_guess()
        x = x0.copy()
        for _ in range(8):
            u0 = mpc.make_step(x)
            x = estimator.make_step(simulator.make_step(u0))
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.abs(x[:self.N_X]) <= 5.0 + 1e-9))

    def test_robust_multistage_mpc_over_a_wired_parameter(self):
        """A wired _p must be usable as a robust-MPC uncertainty."""
        nominal = self._configured_mpc()
        scenarios = np.array([[0.4, -0.2], [0.5, -0.2], [0.3, -0.2]])
        robust = self._configured_mpc(n_robust=1, uncertainty=scenarios)
        self.assertEqual(nominal.opt_x_num.cat.shape[0],
                         (5 + 1) * self.N_X + 5 * self.N_U)
        self.assertGreater(robust.opt_x_num.cat.shape[0],
                           nominal.opt_x_num.cat.shape[0],
                           'the scenario tree did not enlarge the NLP')
        x0 = np.zeros((self.model.n_x, 1))
        robust.x0 = x0
        robust.set_initial_guess()
        u0 = robust.make_step(x0)
        self.assertTrue(np.all(np.isfinite(np.array(u0))))
        print('  robust MPC: nominal opt_x=%d -> robust opt_x=%d (3 scenarios)'
              % (nominal.opt_x_num.cat.shape[0], robust.opt_x_num.cat.shape[0]))

    def test_missing_p_fun_is_still_caught_by_setup(self):
        """The builder must not silently bypass do-mpc's own requirement."""
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = 3
        mpc.settings.supress_ipopt_output()
        mpc.set_objective(lterm=ca.sumsqr(self.model.x['x']),
                          mterm=ca.sumsqr(self.model.x['x']))
        mpc.bounds['lower', '_x', 'x'] = -5 * np.ones((self.N_X, 1))
        mpc.bounds['upper', '_x', 'x'] = 5 * np.ones((self.N_X, 1))
        mpc.bounds['lower', '_u', 'u'] = -2.0
        mpc.bounds['upper', '_u', 'u'] = 2.0
        set_constant_tvp(mpc, ref=0.0)          # tvp given, p deliberately not
        with self.assertRaises(Exception) as ctx:
            mpc.setup()
        self.assertIn('set_p_fun', str(ctx.exception))


@unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed (pip install "do-mpc[full]")')
class TestSlicingDaeAndMeasurements(unittest.TestCase):
    """Output slicing, DAE (algebraic) models, and measurement equations."""

    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.RandomState(20260924)

    # Both helpers ravel: the reference from torch is 1-D, and comparing a
    # (n, 1) array against an (n,) array would broadcast into an n x n all-pairs
    # comparison that silently reports a huge "deviation".
    def rhs(self, model, x, u, z=None):
        return dm(model._rhs_fun(x, u,
                                 np.zeros((model.n_z, 1)) if z is None else z,
                                 np.zeros((model.n_tvp, 1)),
                                 np.zeros((model.n_p, 1)),
                                 np.zeros((model.n_w, 1)))).ravel()

    def alg(self, model, x, u, z):
        return dm(model._alg_fun(x, u, z, np.zeros((model.n_tvp, 1)),
                                 np.zeros((model.n_p, 1)),
                                 np.zeros((model.n_w, 1)))).ravel()

    # ------------------------------------------------------------ output slicing
    def test_one_output_slice_drives_two_states(self):
        module = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 4))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('a', 2), ('b', 2)], u_spec=[('u', 1)],
            wiring=[('z', [('x', 'a'), ('x', 'b'), ('u', 'u')])],
            outputs=[((0, slice(0, 2)), 'a'), ((0, slice(2, 4)), 'b')],
            sample_args=(torch.zeros(1, 5),), input_names=['z'])
        self.assertEqual(model.n_x, 4)
        worst = 0.0
        for _ in range(120):
            a = self.rng.randn(2, 1).astype(np.float32)
            b = self.rng.randn(2, 1).astype(np.float32)
            u = self.rng.randn(1, 1).astype(np.float32)
            x = np.vstack([a, b])
            with torch.no_grad():
                ref = module(torch.tensor(np.vstack([x, u]).astype(np.float32)).T).numpy().ravel()
            worst = max(worst, float(np.abs(self.rhs(model, x, u) - ref).max()))
        self.assertLess(worst, TOL, 'deviation %.2e' % worst)
        print('  sliced output -> 2 states: max|rhs_fun - torch| = %.2e' % worst)

    def test_intermediate_node_can_be_used_as_an_output_key(self):
        """Keys are not limited to declared graph outputs.

        The square 4 -> 4 network keeps every intermediate the same width as the
        state, so wiring `x` alone stays consistent.
        """
        module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(),
                                     torch.nn.Linear(4, 4))
        module.eval()
        proto = do_mpc.sysid.torch_module_to_onnx(module, (torch.zeros(1, 4),), ['z'])
        probe = do_mpc.sysid.ONNXConversion(proto)
        probe.convert(z=ca.SX.sym('z', 1, 4))
        middle = [name for name in probe.node_values
                  if name not in probe.output_layers and name != 'z']
        self.assertTrue(middle, 'the graph has no intermediate node to test with')
        width = probe[middle[0]].shape[1]
        self.assertEqual(width, 4)
        model, _ = ann_to_dompc_model(
            proto, x_spec=[('x', width)], u_spec=[],
            wiring=[('z', [('x', 'x')])], outputs=[(middle[0], 'x')])
        self.assertEqual(model.n_x, width)

    def test_resolve_output_accepts_index_name_and_slice(self):
        from do_mpc.sysid import resolve_output
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                     torch.nn.Linear(4, 2))
        module.eval()
        proto = do_mpc.sysid.torch_module_to_onnx(module, (torch.zeros(1, 3),), ['z'])
        conv = do_mpc.sysid.ONNXConversion(proto)
        conv.convert(z=ca.SX.sym('z', 1, 3))
        by_index = resolve_output(conv, 0)
        by_name = resolve_output(conv, by_index[0])
        self.assertEqual(by_index[0], by_name[0])
        _, sliced = resolve_output(conv, (0, slice(0, 1)))
        self.assertEqual(sliced.shape[1], 1)
        with self.assertRaises(Exception):
            resolve_output(conv, 99)
        with self.assertRaises(Exception):
            resolve_output(conv, 'no_such_node')
        with self.assertRaises(Exception) as ctx:
            resolve_output(conv, (0, slice(5, 9)))
        self.assertIn('selects nothing', str(ctx.exception))

    # ------------------------------------------------------------- measurements
    def test_measurement_from_a_network_head(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 3))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('z', [('x', 'x'), ('u', 'u')])],
            outputs=[((0, slice(0, 2)), 'x')],
            measurements=[((0, slice(2, 3)), 'y_head', True)],
            sample_args=(torch.zeros(1, 3),), input_names=['z'])
        self.assertEqual(model.n_y, 1)
        self.assertEqual(model.n_v, 1)          # meas_noise=True creates the _v
        self.assertIn('y_head', list(model.y.keys()))

    def test_measurement_from_an_existing_state(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 2))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('z', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            measurements=[(('x', 'x'), 'y_full', True)],
            sample_args=(torch.zeros(1, 3),), input_names=['z'])
        self.assertEqual(model.n_y, 2)
        self.assertEqual(model.n_v, 2)

    def test_measurement_without_noise_creates_no_v(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 2))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('z', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            measurements=[(('x', 'x'), 'y_clean', False)],
            sample_args=(torch.zeros(1, 3),), input_names=['z'])
        self.assertEqual(model.n_y, 2)
        self.assertEqual(model.n_v, 0)

    def test_mhe_runs_on_an_ann_model_with_a_measurement(self):
        """The point of set_meas: an estimator can use a partial observation."""
        module = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 2))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)],
            wiring=[('z', [('x', 'x'), ('u', 'u')])], outputs=[(0, 'x')],
            measurements=[(('x', 'x'), 'y_full', True)],
            sample_args=(torch.zeros(1, 3),), input_names=['z'])
        mhe = do_mpc.estimator.MHE(model, [])
        mhe.settings.n_horizon = 4
        mhe.settings.t_step = 0.1
        mhe.settings.supress_ipopt_output()
        mhe.settings.meas_from_data = True
        mhe.set_default_objective(1e-4 * np.eye(model.n_x), np.eye(model.n_v))
        mhe.setup()
        mhe.x0 = np.zeros((model.n_x, 1))
        mhe.set_initial_guess()
        estimate = mhe.make_step(np.zeros((model.n_y, 1)))
        self.assertTrue(np.all(np.isfinite(np.array(estimate))))

    # ---------------------------------------------------------------------- DAE
    def _dae_model(self):
        """A network taking [x; u; z] and returning [x_next; residual]."""
        module = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 4))
        module.eval()
        model, _ = ann_to_dompc_model(
            module, x_spec=[('x', 2)], u_spec=[('u', 1)], z_spec=[('alg', 2)],
            wiring=[('z', [('x', 'x'), ('u', 'u'), ('z', 'alg')])],
            outputs=[((0, slice(0, 2)), 'x')],
            algebraic=[((0, slice(2, 4)), 'alg')],
            sample_args=(torch.zeros(1, 5),), input_names=['z'])
        return model, module

    def test_dae_model_declares_alg_fun(self):
        model, module = self._dae_model()
        self.assertEqual(model.n_z, 2)
        self.assertTrue(hasattr(model, '_alg_fun'))
        worst_rhs = worst_alg = 0.0
        for _ in range(120):
            x = self.rng.randn(2, 1).astype(np.float32)
            u = self.rng.randn(1, 1).astype(np.float32)
            z = self.rng.randn(2, 1).astype(np.float32)
            with torch.no_grad():
                ref = module(torch.tensor(np.vstack([x, u, z]).astype(np.float32)).T).numpy().ravel()
            worst_rhs = max(worst_rhs, float(np.abs(self.rhs(model, x, u, z) - ref[:2]).max()))
            worst_alg = max(worst_alg, float(np.abs(self.alg(model, x, u, z) - ref[2:]).max()))
        self.assertLess(worst_rhs, TOL, 'rhs deviation %.2e' % worst_rhs)
        self.assertLess(worst_alg, TOL, 'alg deviation %.2e' % worst_alg)
        print('  DAE: max|rhs_fun - torch| = %.2e   max|alg_fun - torch| = %.2e'
              % (worst_rhs, worst_alg))

    def test_dae_closed_loop_with_idas(self):
        model, _ = self._dae_model()
        simulator = do_mpc.simulator.Simulator(model)
        simulator.settings.t_step = 0.1
        simulator.settings.integration_tool = 'idas'      # DAE needs IDAS, not CVODES
        simulator.setup()
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.t_step = 0.1
        mpc.settings.n_horizon = 4
        mpc.settings.supress_ipopt_output()
        mpc.set_objective(lterm=ca.sumsqr(model.x['x']), mterm=ca.sumsqr(model.x['x']))
        mpc.set_rterm(u=0.01)
        mpc.bounds['lower', '_x', 'x'] = -5 * np.ones((2, 1))
        mpc.bounds['upper', '_x', 'x'] = 5 * np.ones((2, 1))
        mpc.bounds['lower', '_u', 'u'] = -2.0
        mpc.bounds['upper', '_u', 'u'] = 2.0
        mpc.setup()
        x0 = np.zeros((model.n_x, 1))
        x0[0, 0] = 0.3
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()
        simulator.set_initial_guess()
        estimator = do_mpc.estimator.StateFeedback(model)
        x = x0.copy()
        for _ in range(5):
            u0 = mpc.make_step(x)
            x = estimator.make_step(simulator.make_step(u0))
        self.assertTrue(np.all(np.isfinite(x)), 'DAE closed loop produced non-finite states')
        print('  DAE closed loop (idas): 5 steps, final x[0] = %+.6f' % x[0, 0])

    def test_algebraic_without_equation_is_rejected_at_build_time(self):
        module = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 4))
        module.eval()
        with self.assertRaises(Exception) as ctx:
            ann_to_dompc_model(
                module, x_spec=[('x', 2)], u_spec=[('u', 1)], z_spec=[('alg', 2)],
                wiring=[('z', [('x', 'x'), ('u', 'u'), ('z', 'alg')])],
                outputs=[((0, slice(0, 2)), 'x')],
                algebraic=[],                      # <- missing on purpose
                sample_args=(torch.zeros(1, 5),), input_names=['z'])
        self.assertIn('exactly one equation', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
