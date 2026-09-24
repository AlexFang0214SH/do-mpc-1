#
#   This file is part of do-mpc
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.

"""Regression tests for do_mpc.sysid.ONNXConversion.

Two groups of tests:

* ``TestONNXOperations`` / ``TestONNXConversionGraph`` build tiny ONNX graphs by
  hand with ``onnx.helper``.  They need no deep-learning framework and exercise
  the individual operators, the empty-string optional-input handling and the
  multi-output node handling directly.

* ``TestRecurrentCells`` exports real PyTorch recurrent cells and compares the
  converted CasADi expression against PyTorch numerically.  These are skipped
  when torch is not installed.

Deliberately no golden ``.pkl`` files: the numeric reference is PyTorch itself,
which keeps the tests independent of the installed torch version's RNG stream.
"""

import copy
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

do_mpc_path = '../'
if do_mpc_path not in sys.path:
    sys.path.append(do_mpc_path)

import casadi as ca
import onnx
from onnx import TensorProto, helper, numpy_helper

import do_mpc
from do_mpc.sysid import ONNXConversion, ONNXOperations

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

N_IN, N_HID = 2, 8
OPSET = 17


def dm(value):
    """CasADi DM -> numpy array (avoids the casadi>=3.8 numpy-legacy warning)."""
    return np.array(ca.DM(value).full())


def make_model(nodes, inputs, outputs, name='test_graph', check=True, initializers=None):
    """Assemble an ONNX model. ``check=False`` for deliberately invalid graphs.

    Initializers must be handed to ``make_graph`` up front: the ONNX checker
    verifies topological ordering and treats a not-yet-declared initializer as an
    unknown node input.
    """
    graph = helper.make_graph(nodes, name, inputs, outputs,
                              initializer=list(initializers or []))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)])
    model.ir_version = 9
    if check:
        onnx.checker.check_model(model)
    return model


def tensor_attribute(name, value):
    """Build a TENSOR attribute (make_attribute cannot infer it from a ndarray)."""
    return helper.make_attribute(name, numpy_helper.from_array(
        np.asarray(value, dtype=np.float32), name=name + '_value'))


def tensor_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


class TestONNXOperations(unittest.TestCase):
    """Unit tests for the operators added / fixed in ONNXOperations."""

    def setUp(self):
        self.ops = ONNXOperations()

    def test_identity(self):
        x = ca.SX.sym('x', 2, 3)
        self.assertTrue(ca.is_equal(self.ops.Identity(x), x))

    def test_constant_tensor(self):
        value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        attr = [tensor_attribute('value', value)]
        out = self.ops.Constant(attribute=attr)
        np.testing.assert_allclose(np.asarray(out), value)

    def test_constant_scalars(self):
        self.assertEqual(self.ops.Constant(attribute=[helper.make_attribute('value', 7)]), 7)
        self.assertEqual(self.ops.Constant(attribute=[helper.make_attribute('value', 2.5)]), 2.5)
        np.testing.assert_allclose(
            self.ops.Constant(attribute=[helper.make_attribute('value', [1, 2, 3])]), [1, 2, 3])

    def test_rejected_model_error_does_not_advertise_a_nonexistent_flag(self):
        """The message used to tell users to pass ``from_keras=``, which never existed.

        Naming ``tf2onnx.convert.from_keras(...)`` as the export route is correct;
        advertising a ``from_keras`` *keyword of ONNXConversion* is not.
        """
        with self.assertRaises(Exception) as ctx:
            ONNXConversion('not an onnx model')
        message = str(ctx.exception)
        self.assertNotIn('from_keras flag', message)
        self.assertNotIn('from_keras=', message)
        self.assertIn('onnx.ModelProto', message)
        self.assertIn('onnx.load', message)

    def test_constant_without_value_raises(self):
        with self.assertRaises(Exception):
            self.ops.Constant(attribute=[helper.make_attribute('other', 1)])

    def test_transpose_2d(self):
        x = ca.SX.sym('x', 2, 3)
        out = self.ops.Transpose(x, attribute=[helper.make_attribute('perm', [1, 0])])
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_transpose_default_perm(self):
        x = ca.SX.sym('x', 2, 3)
        self.assertEqual(tuple(self.ops.Transpose(x).shape), (3, 2))

    def test_transpose_identity_perm_rank2(self):
        x = ca.SX.sym('x', 2, 3)
        out = self.ops.Transpose(x, attribute=[helper.make_attribute('perm', [0, 1])])
        self.assertTrue(ca.is_equal(out, x))

    def test_transpose_negative_perm_rank2(self):
        x = ca.SX.sym('x', 2, 3)
        out = self.ops.Transpose(x, attribute=[helper.make_attribute('perm', [-1, -2])])
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_transpose_perm_longer_than_the_known_rank_raises(self):
        """Without rank information a 3-element perm on a 2-D value is invalid.

        The rank-3 cases are covered at graph level, where shape inference
        supplies the offset -- see TestRankTracking.
        """
        x = ca.SX.sym('x', 2, 3)
        for perm in ([1, 0, 2], [0, 1, 2]):
            with self.assertRaises(Exception) as ctx:
                self.ops.Transpose(x, attribute=[helper.make_attribute('perm', perm)])
            self.assertIn('perm', str(ctx.exception))

    def test_unsqueeze_from_axes_input(self):
        """Opset >= 13 passes the axes as a second input rather than an attribute."""
        x = ca.DM(np.arange(6).reshape(1, 6))
        out = self.ops.Unsqueeze(x, axes=np.array([0]))
        # ONNX shape would be (1,1,6); CasADi is 2-D so the singleton is dropped.
        self.assertEqual(tuple(out.shape), (1, 6))
        np.testing.assert_allclose(dm(out), dm(x))

    def test_unsqueeze_from_attribute(self):
        x = ca.DM(np.arange(6).reshape(2, 3))
        out = self.ops.Unsqueeze(x, attribute=[helper.make_attribute('axes', [1])])
        # ONNX shape (2,1,3) -> CasADi (2,3), order preserved
        self.assertEqual(tuple(out.shape), (2, 3))
        np.testing.assert_allclose(dm(out), dm(x))

    def test_unsqueeze_raises_when_no_singleton_can_be_dropped(self):
        x = ca.DM(np.arange(24).reshape(2, 12))
        with self.assertRaises(Exception) as ctx:
            # target shape (2,12,1) has no singleton besides the new one, and
            # after dropping it we are back to rank 2 -> that is fine; force a
            # genuinely unrepresentable case instead:
            self.ops._to_casadi_shape(x, [2, 3, 4], 2, 'Unsqueeze')
        self.assertIn('two-dimensional', str(ctx.exception))

    def test_to_casadi_shape_ranks(self):
        x = ca.DM(np.arange(6).reshape(2, 3))
        self.assertEqual(tuple(self.ops._to_casadi_shape(x, [2, 3], 2, 't').shape), (2, 3))
        self.assertEqual(tuple(self.ops._to_casadi_shape(x, [6], 2, 't').shape), (6, 1))
        self.assertEqual(tuple(self.ops._to_casadi_shape(x, [1, 2, 3], 2, 't').shape), (2, 3))
        self.assertEqual(tuple(self.ops._to_casadi_shape(x, [6, 1, 1], 2, 't').shape), (6, 1))
        # a scalar target is only representable when there is exactly one element
        one = ca.DM(np.array([[3.0]]))
        self.assertEqual(tuple(self.ops._to_casadi_shape(one, [], 2, 't').shape), (1, 1))

    def test_to_casadi_shape_element_count_mismatch_raises(self):
        """Must raise a clear message, not a raw CasADi sparsity assertion."""
        x = ca.DM(np.arange(6).reshape(2, 3))
        with self.assertRaises(Exception) as ctx:
            self.ops._to_casadi_shape(x, [], 2, 'Squeeze')
        self.assertIn('number of elements', str(ctx.exception))

    # -- Slice ---------------------------------------------------------------
    # Since ONNX opset 10 starts/ends/axes/steps are INPUTS, not attributes.
    # Every current exporter (torch.onnx, tf2onnx) produces that form, so the
    # attribute-only implementation used to raise IndexError on all of them.
    def _slice(self, data, starts, ends, axes=None, steps=None, use_attributes=False):
        if use_attributes:
            attribute = [helper.make_attribute('starts', [int(v) for v in starts]),
                         helper.make_attribute('ends', [int(v) for v in ends])]
            if axes is not None:
                attribute.append(helper.make_attribute('axes', [int(v) for v in axes]))
            if steps is not None:
                attribute.append(helper.make_attribute('steps', [int(v) for v in steps]))
            return self.ops.Slice(data, attribute=attribute)
        return self.ops.Slice(data, np.array(starts, dtype=np.int64),
                              np.array(ends, dtype=np.int64),
                              None if axes is None else np.array(axes, dtype=np.int64),
                              None if steps is None else np.array(steps, dtype=np.int64))

    def test_slice_opset10_inputs(self):
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [1], [3], [1]))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[:, 1:3])

    def test_slice_opset9_attributes(self):
        """The legacy attribute form must keep working."""
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [1], [3], [1], use_attributes=True))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[:, 1:3])

    def test_slice_negative_index_and_int_max_sentinel(self):
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [-2], [np.iinfo(np.int64).max], [1]))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[:, -2:])

    def test_slice_with_steps(self):
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [0], [5], [1], [2]))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[:, 0:5:2])

    def test_slice_negative_step(self):
        """starts=3, ends=0, step=-1 selects indices 3, 2, 1 (three elements)."""
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [3], [0], [1], [-1]))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[:, 3:0:-1])

    def test_slice_axes_default_to_range(self):
        """When axes is omitted ONNX defines it as range(len(starts))."""
        data = ca.DM(np.arange(20).reshape(4, 5))
        out = np.asarray(self._slice(data, [0, 1], [2, 4]))
        np.testing.assert_allclose(out, np.arange(20).reshape(4, 5)[0:2, 1:4])

    def test_slice_rank3_raises_clear_error(self):
        with self.assertRaises(Exception) as ctx:
            self.ops.Slice(np.zeros((2, 3, 4)), np.array([0]), np.array([1]), np.array([0]))
        self.assertIn('two-dimensional', str(ctx.exception))

    def test_slice_zero_step_raises(self):
        with self.assertRaises(Exception) as ctx:
            self._slice(ca.DM(np.arange(20).reshape(4, 5)), [0], [2], [1], [0])
        self.assertIn('steps', str(ctx.exception))

    def test_slice_missing_starts_raises_clear_error(self):
        """Must not surface as a bare IndexError."""
        with self.assertRaises(Exception) as ctx:
            self.ops.Slice(ca.DM(np.arange(6).reshape(2, 3)), None, None, attribute=[])
        message = str(ctx.exception)
        self.assertIn('starts', message)
        self.assertNotIsInstance(ctx.exception, IndexError)

    def test_slice_inconsistent_lengths_raise(self):
        with self.assertRaises(Exception) as ctx:
            self.ops.Slice(ca.DM(np.arange(20).reshape(4, 5)),
                           np.array([0, 1]), np.array([2]), np.array([0, 1]))
        self.assertIn('inconsistent', str(ctx.exception))

    # -- Concat --------------------------------------------------------------
    def test_concat_axis_absent_defaults_to_zero(self):
        """ONNX defaults axis to 0; the old code did attribute[0].i -> IndexError."""
        a, b = ca.SX.sym('a', 2, 3), ca.SX.sym('b', 2, 3)
        self.assertEqual(tuple(self.ops.Concat(a, b, attribute=[]).shape), (4, 3))

    def test_concat_axis_variants(self):
        a, b = ca.SX.sym('a', 2, 3), ca.SX.sym('b', 2, 3)
        expected = {0: (4, 3), 1: (2, 6), -1: (2, 6), -2: (4, 3)}
        for axis, shape in expected.items():
            out = self.ops.Concat(a, b, attribute=[helper.make_attribute('axis', axis)])
            self.assertEqual(tuple(out.shape), shape, 'axis=%d' % axis)

    def test_concat_axis_out_of_range_raises(self):
        """axis=2 on a rank-2 tensor is invalid, not a silent wrap onto axis 0."""
        a, b = ca.SX.sym('a', 2, 3), ca.SX.sym('b', 2, 3)
        with self.assertRaises(Exception) as ctx:
            self.ops.Concat(a, b, attribute=[helper.make_attribute('axis', 2)])
        message = str(ctx.exception)
        self.assertTrue('out of range' in message or 'Concat' in message, message)

    # -- Reshape -------------------------------------------------------------
    def test_reshape_plain(self):
        out = np.asarray(self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), np.array([2, 6])))
        self.assertEqual(out.shape, (2, 6))
        np.testing.assert_allclose(out.ravel(), np.arange(12))

    def test_reshape_minus_one_is_inferred(self):
        out = np.asarray(self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), np.array([-1, 6])))
        self.assertEqual(out.shape, (2, 6))

    def test_reshape_zero_copies_input_dim(self):
        out = np.asarray(self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), np.array([0, 4])))
        self.assertEqual(out.shape, (3, 4))

    def test_reshape_to_rank_one_becomes_column(self):
        out = np.asarray(self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), np.array([12])))
        self.assertEqual(out.shape, (12, 1))

    def test_reshape_inconsistent_element_count_raises(self):
        with self.assertRaises(Exception) as ctx:
            self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), np.array([5, 5]))
        self.assertIn('elements', str(ctx.exception))

    def test_reshape_without_shape_input_raises(self):
        with self.assertRaises(Exception):
            self.ops.Reshape(ca.DM(np.arange(12).reshape(3, 4)), None)

    # -- Shape ---------------------------------------------------------------
    def test_shape_returns_a_plain_tuple(self):
        """Documented behaviour: meta-information, not a CasADi value."""
        self.assertEqual(self.ops.Shape(ca.DM(np.zeros((3, 4)))), (3, 4))


class TestExtendedOperators(unittest.TestCase):
    """Unit tests for the element-wise / normalisation / reduction operators.

    References are plain numpy (and scipy for erf), so these need no torch and
    run in CI.
    """

    def setUp(self):
        self.ops = ONNXOperations()
        self.rng = np.random.RandomState(31337)
        self.x = (self.rng.randn(2, 5) * 1.5).astype(np.float64)

    # -- activations ---------------------------------------------------------
    def test_leaky_relu_default_alpha(self):
        got = np.asarray(self.ops.LeakyRelu(ca.DM(self.x)))
        np.testing.assert_allclose(got, np.where(self.x >= 0, self.x, 0.01 * self.x))

    def test_leaky_relu_explicit_alpha(self):
        got = np.asarray(self.ops.LeakyRelu(ca.DM(self.x),
                                            attribute=[helper.make_attribute('alpha', 0.3)]))
        np.testing.assert_allclose(got, np.where(self.x >= 0, self.x, 0.3 * self.x))

    def test_softplus(self):
        got = np.asarray(self.ops.Softplus(ca.DM(self.x)))
        np.testing.assert_allclose(got, np.log1p(np.exp(self.x)), rtol=1e-12)

    def test_softmax_along_columns(self):
        for axis in (-1, 1):
            got = np.asarray(self.ops.Softmax(ca.DM(self.x),
                                              attribute=[helper.make_attribute('axis', axis)]))
            np.testing.assert_allclose(got.sum(axis=1), np.ones(2), atol=1e-12)
            reference = np.exp(self.x - self.x.max(axis=1, keepdims=True))
            np.testing.assert_allclose(got, reference / reference.sum(axis=1, keepdims=True),
                                       rtol=1e-12)

    def test_softmax_along_rows(self):
        got = np.asarray(self.ops.Softmax(ca.DM(self.x),
                                          attribute=[helper.make_attribute('axis', 0)]))
        np.testing.assert_allclose(got.sum(axis=0), np.ones(5), atol=1e-12)

    def test_softmax_is_overflow_safe(self):
        """Subtracting the row max must keep large inputs finite."""
        big = np.array([[1000.0, 1001.0, 999.0]])
        got = np.asarray(self.ops.Softmax(ca.DM(big)))
        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_allclose(got.sum(), 1.0, atol=1e-12)

    def test_clip_with_inputs(self):
        got = np.asarray(self.ops.Clip(ca.DM(self.x), -0.5, 0.5))
        np.testing.assert_allclose(got, np.clip(self.x, -0.5, 0.5))

    def test_clip_with_attributes(self):
        """opset <= 10 carries the bounds as attributes."""
        got = np.asarray(self.ops.Clip(ca.DM(self.x), None, None,
                                       attribute=[helper.make_attribute('min', -0.5),
                                                  helper.make_attribute('max', 0.5)]))
        np.testing.assert_allclose(got, np.clip(self.x, -0.5, 0.5))

    def test_clip_one_sided(self):
        got = np.asarray(self.ops.Clip(ca.DM(self.x), 0.0, None))
        np.testing.assert_allclose(got, np.maximum(self.x, 0.0))

    # -- arithmetic ----------------------------------------------------------
    def test_elementwise_arithmetic_matches_numpy(self):
        from scipy.special import erf as scipy_erf
        y = (self.rng.randn(2, 5) * 0.4 + 2.0).astype(np.float64)   # positive, for sqrt/log
        d = ca.DM(self.x)
        e = ca.DM(y)
        cases = [
            ('Div', self.ops.Div(d, e), self.x / y),
            ('Pow', self.ops.Pow(ca.DM(np.abs(self.x)), e), np.abs(self.x) ** y),
            ('Sqrt', self.ops.Sqrt(e), np.sqrt(y)),
            ('Erf', self.ops.Erf(d), scipy_erf(self.x)),
            ('Abs', self.ops.Abs(d), np.abs(self.x)),
            ('Neg', self.ops.Neg(d), -self.x),
            ('Exp', self.ops.Exp(d), np.exp(self.x)),
            ('Log', self.ops.Log(e), np.log(y)),
        ]
        for name, got, expected in cases:
            np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-11,
                                       err_msg=name)

    # -- shape ---------------------------------------------------------------
    def test_flatten_axis_default_is_a_noop_for_2d(self):
        got = np.asarray(self.ops.Flatten(ca.DM(self.x)))
        self.assertEqual(got.shape, (2, 5))
        np.testing.assert_allclose(got, self.x)

    def test_flatten_axis_zero_flattens_to_a_row(self):
        got = np.asarray(self.ops.Flatten(ca.DM(self.x),
                                          attribute=[helper.make_attribute('axis', 0)]))
        self.assertEqual(got.shape, (1, 10))
        np.testing.assert_allclose(got.ravel(), self.x.ravel())

    # -- normalisation -------------------------------------------------------
    def test_batch_normalization_inference(self):
        scale = self.rng.randn(5).astype(np.float64)
        bias = self.rng.randn(5).astype(np.float64)
        mean = self.rng.randn(5).astype(np.float64)
        var = np.abs(self.rng.randn(5)).astype(np.float64) + 0.1
        got = np.asarray(self.ops.BatchNormalization(
            ca.DM(self.x), scale, bias, mean, var,
            attribute=[helper.make_attribute('epsilon', 1e-5)]))
        expected = (self.x - mean.reshape(1, -1)) / np.sqrt(var.reshape(1, -1) + 1e-5) \
            * scale.reshape(1, -1) + bias.reshape(1, -1)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)

    def test_batch_normalization_rejects_training_mode(self):
        with self.assertRaises(Exception) as ctx:
            self.ops.BatchNormalization(ca.DM(self.x), np.ones(5), np.zeros(5),
                                        np.zeros(5), np.ones(5),
                                        attribute=[helper.make_attribute('training_mode', 1)])
        self.assertIn('training mode', str(ctx.exception))

    def test_layer_normalization(self):
        scale = self.rng.randn(5).astype(np.float64)
        bias = self.rng.randn(5).astype(np.float64)
        got = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, bias,
            attribute=[helper.make_attribute('epsilon', 1e-5),
                       helper.make_attribute('axis', -1)]))
        mu = self.x.mean(axis=1, keepdims=True)
        centred = self.x - mu
        var = (centred ** 2).mean(axis=1, keepdims=True)
        expected = centred / np.sqrt(var + 1e-5) * scale.reshape(1, -1) + bias.reshape(1, -1)
        # atol is needed: individual entries can be arbitrarily close to zero,
        # where a pure relative tolerance is meaningless.
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)

    def test_layer_normalization_without_bias(self):
        got = np.asarray(self.ops.LayerNormalization(ca.DM(self.x), np.ones(5), None))
        mu = self.x.mean(axis=1, keepdims=True)
        centred = self.x - mu
        var = (centred ** 2).mean(axis=1, keepdims=True)
        np.testing.assert_allclose(got, centred / np.sqrt(var + 1e-5), rtol=1e-11)

    def test_layer_normalization_over_both_axes(self):
        """ONNX normalises over ``[axis .. rank-1]``, so axis=0 covers everything.

        ``torch.nn.LayerNorm([rows, cols])`` exports exactly this graph. Taking
        the statistics per row -- what ``axis=0`` used to do -- is silently wrong.
        """
        scale = self.rng.randn(*self.x.shape).astype(np.float64)
        bias = self.rng.randn(*self.x.shape).astype(np.float64)
        got = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, bias,
            attribute=[helper.make_attribute('epsilon', 1e-5),
                       helper.make_attribute('axis', 0)]))
        mean = self.x.mean()
        variance = ((self.x - mean) ** 2).mean()
        expected = (self.x - mean) / np.sqrt(variance + 1e-5) * scale + bias
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)
        # Guard against a regression to the per-row interpretation.
        row_mean = self.x.mean(axis=1, keepdims=True)
        row_var = ((self.x - row_mean) ** 2).mean(axis=1, keepdims=True)
        per_row = (self.x - row_mean) / np.sqrt(row_var + 1e-5) * scale + bias
        self.assertGreater(np.abs(got - per_row).max(), 1e-3)

    def test_layer_normalization_over_both_axes_without_bias(self):
        scale = self.rng.randn(*self.x.shape).astype(np.float64)
        got = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, None, attribute=[helper.make_attribute('axis', 0)]))
        mean = self.x.mean()
        variance = ((self.x - mean) ** 2).mean()
        np.testing.assert_allclose(got, (self.x - mean) / np.sqrt(variance + 1e-5) * scale,
                                   rtol=1e-11, atol=1e-13)

    def test_layer_normalization_axis_minus_two_matches_axis_zero(self):
        scale = np.ones_like(self.x)
        got = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, None, attribute=[helper.make_attribute('axis', -2)]))
        expected = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, None, attribute=[helper.make_attribute('axis', 0)]))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_layer_normalization_last_axis_is_still_a_single_axis_reduction(self):
        """axis=-1 must not be flattened -- it is the common exporter case."""
        scale = self.rng.randn(self.x.shape[1]).astype(np.float64)
        bias = self.rng.randn(self.x.shape[1]).astype(np.float64)
        got = np.asarray(self.ops.LayerNormalization(
            ca.DM(self.x), scale, bias,
            attribute=[helper.make_attribute('epsilon', 1e-5),
                       helper.make_attribute('axis', -1)]))
        mean = self.x.mean(axis=1, keepdims=True)
        variance = ((self.x - mean) ** 2).mean(axis=1, keepdims=True)
        expected = (self.x - mean) / np.sqrt(variance + 1e-5) * scale + bias
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)

    def test_as_matrix_keeps_the_row_column_layout(self):
        value = np.arange(6, dtype=np.float64).reshape(2, 3)
        np.testing.assert_allclose(np.asarray(ONNXOperations._as_matrix(value, 2, 3)), value)
        np.testing.assert_allclose(
            np.asarray(ONNXOperations._as_matrix(ca.DM(value), 2, 3)), value)

    # -- reduction -----------------------------------------------------------
    def test_reduce_mean_and_sum_both_axis_forms(self):
        for axis in (0, 1, -1, -2):
            normalised = axis % 2
            attribute = [helper.make_attribute('axes', [axis])]
            mean = np.asarray(self.ops.ReduceMean(ca.DM(self.x), None, attribute))
            total = np.asarray(self.ops.ReduceSum(ca.DM(self.x), None, attribute))
            np.testing.assert_allclose(mean.ravel(), self.x.mean(axis=normalised), rtol=1e-12)
            np.testing.assert_allclose(total.ravel(), self.x.sum(axis=normalised), rtol=1e-12)

    def test_reduce_axes_as_input_opset18(self):
        got = np.asarray(self.ops.ReduceMean(ca.DM(self.x), np.array([1], dtype=np.int64), None))
        np.testing.assert_allclose(got.ravel(), self.x.mean(axis=1), rtol=1e-12)

    def test_reduce_over_several_axes_raises(self):
        with self.assertRaises(Exception) as ctx:
            self.ops.ReduceMean(ca.DM(self.x), np.array([0, 1], dtype=np.int64), None)
        self.assertIn('two-dimensional', str(ctx.exception))

    # -- helpers -------------------------------------------------------------
    def test_as_row_reshapes_per_channel_params(self):
        self.assertEqual(np.asarray(self.ops._as_row(np.arange(5))).shape, (1, 5))
        self.assertEqual(tuple(self.ops._as_row(ca.DM(np.arange(5).reshape(5, 1))).shape), (1, 5))

    def test_reduce_after_unsqueeze_follows_onnx_axis_semantics(self):
        """Rank tracking: a positive axis is remapped through the dropped dims.

        ``Unsqueeze`` raises a (B, C) tensor to the ONNX shape [1, B, C], which
        CasADi can only hold as (B, C). ``ONNXConversion`` runs
        ``onnx.shape_inference`` and records that one leading dimension was
        dropped, so ``ReduceMean(axis=1)`` correctly reduces over B -- the ONNX
        axis 1 of [1, B, C] -- rather than over CasADi's columns.

        This used to return the CasADi column mean; the assertion below is the
        ONNX-correct answer and is the regression guard for the rank tracking.
        """
        initializers = [numpy_helper.from_array(np.array([0], dtype=np.int64), 'unsq')]
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 # axes is an attribute up to opset 17 and an input from opset 18
                 helper.make_node('ReduceMean', ['u3'], ['y'], axes=[1])]
        model = make_model(nodes, [tensor_info('x', (2, 3))],
                           [tensor_info('y', (1, 3))], initializers=initializers)
        value = np.arange(6, dtype=np.float64).reshape(2, 3)
        conv = ONNXConversion(model)
        conv.convert(x=value)
        self.assertEqual(conv.ranks.get('u3'), 3,
                         'shape inference did not record the rank-3 intermediate')
        got = np.asarray(conv['y'])
        expected = value.reshape(1, 2, 3).mean(axis=1)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected)
        # and it must NOT be the CasADi column mean, which was the old wrong
        # answer. Shapes differ, so compare defensively instead of with allclose.
        old_wrong_answer = value.mean(axis=1)
        same_as_old = (got.shape == old_wrong_answer.shape
                       and np.allclose(got, old_wrong_answer))
        self.assertFalse(same_as_old, 'regressed to the CasADi column mean')

    def test_negative_axis_is_safe_after_rank_reduction(self):
        """The common exporter convention (axis=-1) is unaffected by the above."""
        initializers = [numpy_helper.from_array(np.array([0], dtype=np.int64), 'unsq')]
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('ReduceMean', ['u3'], ['y'], axes=[-1])]
        model = make_model(nodes, [tensor_info('x', (2, 3))],
                           [tensor_info('y', (2, 1))], initializers=initializers)
        value = np.arange(6, dtype=np.float64).reshape(2, 3)
        conv = ONNXConversion(model)
        conv.convert(x=value)
        np.testing.assert_allclose(np.asarray(conv['y']).ravel(), value.mean(axis=-1))

    def test_attribute_float_helper(self):
        self.assertEqual(self.ops._attribute_float([helper.make_attribute('epsilon', 0.25)],
                                                   'epsilon', 1e-5), 0.25)
        self.assertEqual(self.ops._attribute_float([], 'epsilon', 1e-5), 1e-5)

    def test_squeeze_removes_axes(self):
        x = ca.DM(np.arange(6).reshape(1, 6))
        self.assertEqual(tuple(self.ops.Squeeze(x, axes=np.array([0])).shape), (6, 1))

    def test_squeeze_is_not_a_silent_noop(self):
        """The old implementation returned args[0] unchanged; verify it does not."""
        x = ca.DM(np.arange(6).reshape(1, 6))
        out = self.ops.Squeeze(x, axes=np.array([0]))
        self.assertNotEqual(tuple(out.shape), tuple(x.shape))

    def test_split_returns_marker(self):
        from do_mpc.sysid._onnxconversion import _SplitRequest
        x = ca.SX.sym('x', 1, 8)
        req = self.ops.Split(x, np.array([2, 3, 3]), attribute=[helper.make_attribute('axis', 1)])
        self.assertIsInstance(req, _SplitRequest)
        self.assertEqual(req.axis, 1)
        self.assertEqual(req.sizes, [2, 3, 3])
        self.assertEqual(req.total, 8)

    def test_split_without_sizes_requests_equal_split(self):
        from do_mpc.sysid._onnxconversion import _SplitRequest
        x = ca.SX.sym('x', 6, 1)
        req = self.ops.Split(x, None)
        self.assertIsInstance(req, _SplitRequest)
        self.assertIsNone(req.sizes)
        self.assertEqual(req.axis, 0)
        self.assertEqual(req.total, 6)


class TestONNXConversionGraph(unittest.TestCase):
    """Graph-level tests built with onnx.helper (no torch required)."""

    def test_constant_then_add(self):
        const = np.array([[1.0, 2.0]], dtype=np.float32)
        nodes = [
            helper.make_node('Constant', inputs=[], outputs=['c'],
                             value=numpy_helper.from_array(const, name='cval')),
            helper.make_node('Add', inputs=['x', 'c'], outputs=['y']),
        ]
        model = make_model(nodes, [tensor_info('x', (1, 2))], [tensor_info('y', (1, 2))])
        conv = ONNXConversion(model)
        # CasADi >= 3.8 matches symbols by identity, not by name: the very same
        # SX object has to be used for convert() and for ca.Function().
        x_sym = ca.SX.sym('x', 1, 2)
        conv.convert(x=x_sym)
        f = ca.Function('f', [x_sym], [conv['y']])
        np.testing.assert_allclose(dm(f(np.array([[10.0, 20.0]]))), [[11.0, 22.0]])

    def test_split_with_explicit_sizes_registers_all_outputs(self):
        """A multi-output node must register EVERY declared output name."""
        nodes = [
            helper.make_node('Constant', inputs=[], outputs=['sizes'],
                             value=numpy_helper.from_array(np.array([1, 2, 1]), name='sz')),
            helper.make_node('Split', inputs=['x', 'sizes'], outputs=['a', 'b', 'c'], axis=1),
        ]
        model = make_model(nodes, [tensor_info('x', (1, 4))],
                           [tensor_info('a', (1, 1)), tensor_info('b', (1, 2)),
                            tensor_info('c', (1, 1))])
        conv = ONNXConversion(model)
        conv.convert(x=np.array([[1.0, 2.0, 3.0, 4.0]]))
        for name in ('a', 'b', 'c'):
            self.assertIn(name, conv.node_values, 'output %r was not registered' % name)
        np.testing.assert_allclose(np.asarray(conv['a']).ravel(), [1.0])
        np.testing.assert_allclose(np.asarray(conv['b']).ravel(), [2.0, 3.0])
        np.testing.assert_allclose(np.asarray(conv['c']).ravel(), [4.0])

    def test_split_with_empty_optional_input_splits_equally(self):
        """Empty-string optional input -> None, and an equal-sized split is inferred."""
        nodes = [helper.make_node('Split', inputs=['x', ''], outputs=['a', 'b'], axis=1)]
        model = make_model(nodes, [tensor_info('x', (1, 6))],
                           [tensor_info('a', (1, 3)), tensor_info('b', (1, 3))])
        conv = ONNXConversion(model)
        conv.convert(x=np.arange(6, dtype=np.float32).reshape(1, 6))
        np.testing.assert_allclose(np.asarray(conv['a']).ravel(), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(np.asarray(conv['b']).ravel(), [3.0, 4.0, 5.0])

    def test_split_size_count_mismatch_raises(self):
        nodes = [
            helper.make_node('Constant', inputs=[], outputs=['sizes'],
                             value=numpy_helper.from_array(np.array([2, 2]), name='sz')),
            helper.make_node('Split', inputs=['x', 'sizes'], outputs=['a', 'b', 'c'], axis=1),
        ]
        model = make_model(nodes, [tensor_info('x', (1, 6))],
                           [tensor_info('a', (1, 2)), tensor_info('b', (1, 2)),
                            tensor_info('c', (1, 2))])
        conv = ONNXConversion(model)
        with self.assertRaises(Exception) as ctx:
            conv.convert(x=np.zeros((1, 6), dtype=np.float32))
        self.assertIn('split', str(ctx.exception).lower())

    def test_transpose_roundtrip(self):
        nodes = [
            helper.make_node('Transpose', inputs=['x'], outputs=['t'], perm=[1, 0]),
            helper.make_node('Transpose', inputs=['t'], outputs=['y'], perm=[1, 0]),
        ]
        model = make_model(nodes, [tensor_info('x', (2, 3))], [tensor_info('y', (2, 3))])
        conv = ONNXConversion(model)
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        conv.convert(x=value)
        np.testing.assert_allclose(dm(conv['y']), value)

    def test_unknown_input_reference_raises(self):
        nodes = [helper.make_node('Add', inputs=['x', 'nonexistent'], outputs=['y'])]
        model = make_model(nodes, [tensor_info('x', (1, 2))], [tensor_info('y', (1, 2))],
                           check=False)
        conv = ONNXConversion(model)
        with self.assertRaises(Exception) as ctx:
            conv.convert(x=np.zeros((1, 2), dtype=np.float32))
        self.assertIn('nonexistent', str(ctx.exception))

    def test_unimplemented_op_raises(self):
        nodes = [helper.make_node('Conv', inputs=['x', 'w'], outputs=['y'])]
        model = make_model(nodes, [tensor_info('x', (1, 2))], [tensor_info('y', (1, 2))],
                           check=False)
        model.graph.initializer.append(numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), name='w'))
        conv = ONNXConversion(model)
        with self.assertRaises(Exception) as ctx:
            conv.convert(x=np.zeros((1, 2), dtype=np.float32))
        self.assertIn("Operation 'Conv' not implemented", str(ctx.exception))

    # ---------------------------------------------------------------------
    # End-to-end graph coverage. The hand-built graph below reproduces what
    # examples/tools/onnx_conversion/onnx_conversion_02.py exercises (two
    # inputs, Concat, two branches, Add, a Slice and a final Tanh projection)
    # so the same operator paths are regression-tested in CI without requiring
    # tensorflow or tf2onnx.
    # ---------------------------------------------------------------------
    def _multi_input_graph(self):
        rng = np.random.RandomState(20240921)
        weights = [(rng.randn(5, 5) * 0.5).astype(np.float32),
                   (rng.randn(5, 5) * 0.5).astype(np.float32),
                   (rng.randn(5, 2) * 0.5).astype(np.float32)]
        initializers = [numpy_helper.from_array(np.ascontiguousarray(w.T), n)
                        for w, n in zip(weights, ['W1T', 'W2T', 'W3T'])]
        initializers += [numpy_helper.from_array(np.array(v, dtype=np.int64), n)
                         for v, n in [([0], 'sl_starts'), ([2], 'sl_ends'), ([1], 'sl_axes')]]
        nodes = [
            helper.make_node('Concat', ['first_input', 'second_input'], ['cat'], axis=1),
            helper.make_node('MatMul', ['cat', 'W1T'], ['h1']),
            helper.make_node('Tanh', ['h1'], ['t1']),
            helper.make_node('MatMul', ['t1', 'W2T'], ['h2']),
            helper.make_node('Relu', ['h2'], ['r2']),
            helper.make_node('Add', ['t1', 'r2'], ['s']),
            helper.make_node('Slice', ['s', 'sl_starts', 'sl_ends', 'sl_axes'], ['sl']),
            helper.make_node('MatMul', ['sl', 'W3T'], ['pre_out']),
            helper.make_node('Tanh', ['pre_out'], ['model_output']),
        ]
        model = make_model(nodes,
                           [tensor_info('first_input', (1, 3)), tensor_info('second_input', (1, 2))],
                           [tensor_info('model_output', (1, 5))],
                           initializers=initializers)
        return model, weights

    @staticmethod
    def _numpy_reference(weights, a, b):
        """Exactly the graph above, in numpy. Keep the two in sync."""
        w1, w2, w3 = weights
        cat = np.hstack([a, b])
        t1 = np.tanh(cat @ w1.T)
        r2 = np.maximum(0.0, t1 @ w2.T)
        s = t1 + r2
        return np.tanh(s[:, 0:2] @ w3.T)

    def test_multi_input_graph_with_concat_add_and_slice(self):
        model, weights = self._multi_input_graph()
        conv = ONNXConversion(model)
        self.assertEqual(sorted(conv.inputshape), ['first_input', 'second_input'])

        i1 = ca.SX.sym('in1', 1, 3)
        i2 = ca.SX.sym('in2', 1, 2)
        conv.convert(first_input=i1, second_input=i2)
        f = ca.Function('f', [i1, i2], [conv['model_output']])

        rng = np.random.RandomState(5)
        worst = 0.0
        for _ in range(150):
            a = rng.randn(1, 3).astype(np.float32)
            b = rng.randn(1, 2).astype(np.float32)
            worst = max(worst, float(np.abs(dm(f(a, b)) - self._numpy_reference(weights, a, b)).max()))
        self.assertLess(worst, 1e-5, 'max deviation %.2e' % worst)
        print('  multi-input graph (Concat/MatMul/Tanh/Relu/Add/Slice): max diff = %.2e' % worst)

    def test_multi_input_graph_numeric_and_mixed_inputs(self):
        """The three input modes the old onnx_conversion_02.py exercised."""
        model, weights = self._multi_input_graph()
        rng = np.random.RandomState(11)
        a = rng.randn(1, 3).astype(np.float32)
        b = rng.randn(1, 2).astype(np.float32)
        reference = self._numpy_reference(weights, a, b)

        numeric = ONNXConversion(model)
        numeric.convert(first_input=a, second_input=b)
        np.testing.assert_allclose(np.asarray(numeric['model_output']), reference, atol=1e-5)

        i1 = ca.SX.sym('in1', 1, 3)
        mixed = ONNXConversion(model)
        mixed.convert(first_input=i1, second_input=b)
        f = ca.Function('f', [i1], [mixed['model_output']])
        np.testing.assert_allclose(dm(f(a)), reference, atol=1e-5)

    def test_slice_through_convert_with_opset10_inputs(self):
        """Regression guard: Slice used to read attributes and raise IndexError."""
        initializers = [numpy_helper.from_array(np.array(v, dtype=np.int64), n)
                        for v, n in [([1], 'st'), ([4], 'en'), ([1], 'ax')]]
        nodes = [helper.make_node('Slice', ['x', 'st', 'en', 'ax'], ['y'])]
        model = make_model(nodes, [tensor_info('x', (2, 6))], [tensor_info('y', (2, 3))],
                           initializers=initializers)
        conv = ONNXConversion(model)
        value = np.arange(12, dtype=np.float32).reshape(2, 6)
        conv.convert(x=value)
        np.testing.assert_allclose(np.asarray(conv['y']), value[:, 1:4])

    def test_reshape_preserves_row_major_order_through_convert(self):
        """CasADi reshape is column-major; ONNX Reshape is row-major."""
        initializers = [numpy_helper.from_array(np.array([2, 6], dtype=np.int64), 'shape')]
        nodes = [helper.make_node('Reshape', ['x', 'shape'], ['y'])]
        model = make_model(nodes, [tensor_info('x', (3, 4))], [tensor_info('y', (2, 6))],
                           initializers=initializers)
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        for feed in (value, ca.DM(value)):
            conv = ONNXConversion(model)
            conv.convert(x=feed)
            np.testing.assert_allclose(np.asarray(conv['y']), value.reshape(2, 6))

    def test_concat_without_axis_attribute_is_tolerated(self):
        """Defensive: ONNX requires axis, but attribute[0] must not be assumed.

        The old implementation read ``attribute[0].i`` unconditionally, so any
        node whose first attribute was not ``axis`` (or which omitted it) raised
        IndexError. The graph below is deliberately spec-invalid, hence check=False.
        """
        nodes = [helper.make_node('Concat', ['a', 'b'], ['y'])]
        model = make_model(nodes, [tensor_info('a', (2, 3)), tensor_info('b', (2, 3))],
                           [tensor_info('y', (4, 3))], check=False)
        conv = ONNXConversion(model)
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.arange(6, dtype=np.float32).reshape(2, 3) + 100
        conv.convert(a=a, b=b)
        np.testing.assert_allclose(np.asarray(conv['y']), np.vstack([a, b]))


class TestRecurrentCells(unittest.TestCase):
    """Export real PyTorch recurrent cells and compare against PyTorch numerically.

    The cells are exported rather than the sequence modules on purpose: a cell has
    no fused ONNX counterpart, so torch decomposes it into primitive operators
    that ONNXConversion supports.
    """

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        # Keep temp files inside testing/: on Windows the system temp directory
        # can hit permission errors while torch still holds a handle.
        self.tmp_dir = tempfile.mkdtemp(prefix='_onnx_tmp_', dir=os.getcwd())
        self.rng = np.random.RandomState(1234)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def export(self, module, args, names, stem):
        path = os.path.join(self.tmp_dir, stem + '.onnx')
        torch.onnx.export(module, args, path, input_names=names,
                          dynamo=False, opset_version=OPSET)
        return onnx.load(path)

    def build_function(self, conv, names):
        syms = {n: ca.SX.sym(n, 1, shape[-1]) for n, shape in conv.inputshape.items()}
        conv.convert(**syms)
        return (ca.Function('f', [syms[n] for n in names],
                            [conv[o] for o in conv.output_layers]), syms)

    def random_inputs(self, with_cell_state):
        x = self.rng.randn(1, N_IN).astype(np.float32)
        h = self.rng.randn(1, N_HID).astype(np.float32)
        c = self.rng.randn(1, N_HID).astype(np.float32) if with_cell_state else None
        return x, h, c

    def assert_matches_torch(self, module, names, with_cell_state, n_points=200):
        args = (torch.zeros(1, N_IN),
                (torch.zeros(1, N_HID), torch.zeros(1, N_HID))) if with_cell_state \
            else (torch.zeros(1, N_IN), torch.zeros(1, N_HID))
        conv = ONNXConversion(self.export(module, args, names, names[0]))
        f, _ = self.build_function(conv, names)

        worst = 0.0
        for _ in range(n_points):
            x, h, c = self.random_inputs(with_cell_state)
            with torch.no_grad():
                if with_cell_state:
                    h_t, c_t = module(torch.tensor(x), (torch.tensor(h), torch.tensor(c)))
                    reference = [h_t, c_t]
                    got = f(x, h, c)
                else:
                    reference = [module(torch.tensor(x), torch.tensor(h))]
                    got = f(x, h)
            got = got if isinstance(got, (list, tuple)) else [got]
            self.assertEqual(len(got), len(reference),
                             'converter returned %d outputs, torch returned %d'
                             % (len(got), len(reference)))
            for g, r in zip(got, reference):
                worst = max(worst, float(np.abs(dm(g) - r.numpy()).max()))
        # float32 round-trip through torch -> ONNX -> CasADi (float64)
        self.assertLess(worst, 1e-5, 'max deviation from torch was %.2e' % worst)
        return worst

    def test_lstm_cell(self):
        worst = self.assert_matches_torch(torch.nn.LSTMCell(N_IN, N_HID),
                                          ['x', 'h', 'c'], with_cell_state=True)
        print('LSTMCell max|casadi-torch| = %.2e' % worst)

    def test_gru_cell(self):
        worst = self.assert_matches_torch(torch.nn.GRUCell(N_IN, N_HID),
                                          ['x', 'h'], with_cell_state=False)
        print('GRUCell  max|casadi-torch| = %.2e' % worst)

    def test_rnn_cell_tanh(self):
        worst = self.assert_matches_torch(
            torch.nn.RNNCell(N_IN, N_HID, nonlinearity='tanh'),
            ['x', 'h'], with_cell_state=False)
        print('RNNCell  max|casadi-torch| = %.2e' % worst)

    def test_lstm_cell_has_two_outputs(self):
        """LSTMCell must expose both h_next and c_next (multi-output handling)."""
        module = torch.nn.LSTMCell(N_IN, N_HID)
        conv = ONNXConversion(self.export(
            module, (torch.zeros(1, N_IN), (torch.zeros(1, N_HID), torch.zeros(1, N_HID))),
            ['x', 'h', 'c'], 'lstm'))
        syms = {n: ca.SX.sym(n, 1, s[-1]) for n, s in conv.inputshape.items()}
        conv.convert(**syms)
        self.assertEqual(len(conv.output_layers), 2)
        for name in conv.output_layers:
            self.assertIn(name, conv.node_values)

    def test_feedforward_network(self):
        """Non-recurrent baseline: Gemm + Tanh stack."""
        module = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                     torch.nn.Linear(5, 1))
        onnx_model = self.export(module, torch.zeros(1, 3), ['input'], 'ffn')
        conv = ONNXConversion(onnx_model)
        x_sym = ca.SX.sym('input', 1, 3)
        conv.convert(input=x_sym)
        f = ca.Function('f', [x_sym], [conv[conv.output_layers[0]]])
        worst = 0.0
        for _ in range(200):
            v = self.rng.randn(1, 3).astype(np.float32)
            with torch.no_grad():
                ref = module(torch.tensor(v)).numpy()
            worst = max(worst, float(np.abs(dm(f(v)) - ref).max()))
        self.assertLess(worst, 1e-5)
        print('FFN      max|casadi-torch| = %.2e' % worst)

    def test_sequence_lstm_is_rejected_with_clear_message(self):
        """The fused 3-D LSTM operator cannot be represented; it must fail loudly."""
        module = torch.nn.LSTM(N_IN, N_HID, batch_first=True)
        onnx_model = self.export(module, torch.zeros(1, 4, N_IN), ['x'], 'lstm_seq')
        conv = ONNXConversion(onnx_model)
        with self.assertRaises(Exception) as ctx:
            conv.convert(x=np.zeros((1, 4, N_IN), dtype=np.float32))
        message = str(ctx.exception)
        acceptable = ('two-dimensional', 'not implemented', 'dropped', 'permutation')
        self.assertTrue(any(word in message for word in acceptable),
                        'unexpected error message: %s' % message)


@unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed (pip install "do-mpc[full]")')
class TestFrameworkLayers(unittest.TestCase):
    """Export real torch layers and compare the conversion against torch.

    This is the test that matters most in practice: it exercises whatever op
    combination the exporter actually emits, rather than a hand-picked subset.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.tmp_dir = tempfile.mkdtemp(prefix='_onnx_tmp_', dir=os.getcwd())
        self.rng = np.random.RandomState(99)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def check_layer(self, name, module, shapes, extra_args=None):
        module.eval()
        args = tuple(torch.zeros(*shape) for shape in shapes)
        if extra_args is not None:
            args = extra_args(args)
        names = ['x'] + ['i%d' % k for k in range(len(shapes) - 1)]
        path = os.path.join(self.tmp_dir, name + '.onnx')
        torch.onnx.export(module, args, path, input_names=names,
                          dynamo=False, opset_version=OPSET)
        conv = ONNXConversion(onnx.load(path))
        syms = {n: ca.SX.sym(n, 1, shape[-1] if shape[-1] else 1)
                for n, shape in conv.inputshape.items()}
        conv.convert(**{k: syms[k] for k in conv.inputshape})
        f = ca.Function('f', [syms[k] for k in conv.inputshape],
                        [conv[o] for o in conv.output_layers])

        worst = 0.0
        for _ in range(80):
            values = [(self.rng.randn(*shape) * 0.4).astype(np.float32) for shape in shapes]
            call = [torch.tensor(v) for v in values]
            if extra_args is not None:
                call = extra_args(call, tensors=False)
            with torch.no_grad():
                reference = module(*call)
            got = f(*values)
            got = got if isinstance(got, (list, tuple)) else [got]
            refs = reference if isinstance(reference, tuple) else [reference]
            self.assertEqual(len(got), len(refs), '%s: output count mismatch' % name)
            for g, r in zip(got, refs):
                array = np.asarray(r)
                worst = max(worst, float(np.abs(dm(g).reshape(array.shape) - array).max()))
        self.assertLess(worst, 1e-5, '%s deviates from torch by %.2e' % (name, worst))
        print('  %-14s max|torch-casadi| = %.2e' % (name, worst))

    @staticmethod
    def _stack(hidden):
        def wrap(args, tensors=True):
            if tensors:
                return (args[0], tuple(args[1:]))
            return (args[0], tuple(args[1:]))
        return wrap

    def test_activation_layers(self):
        for name, activation in [('tanh', torch.nn.Tanh), ('sigmoid', torch.nn.Sigmoid),
                                 ('relu', torch.nn.ReLU), ('elu', torch.nn.ELU),
                                 ('leaky_relu', torch.nn.LeakyReLU),
                                 ('gelu', torch.nn.GELU), ('silu', torch.nn.SiLU),
                                 ('softplus', torch.nn.Softplus),
                                 ('hardtanh', torch.nn.Hardtanh),
                                 ('softmax', lambda: torch.nn.Softmax(dim=-1))]:
            module = torch.nn.Sequential(torch.nn.Linear(4, 4), activation())
            self.check_layer(name, module, [(1, 4)])

    def test_normalization_layers(self):
        self.check_layer('batchnorm',
                         torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)),
                         [(1, 4)])
        self.check_layer('layernorm',
                         torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4)),
                         [(1, 4)])
        # LayerNorm over *all* input dimensions exports axis=-2, whose ONNX
        # normalisation group covers both CasADi axes.
        self.check_layer('layernorm_all_dims',
                         torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm([1, 4])),
                         [(1, 4)])

    def test_flatten_and_dropout(self):
        self.check_layer('flatten',
                         torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Flatten()),
                         [(1, 4)])
        # Dropout in eval mode must vanish from the graph entirely.
        self.check_layer('dropout_eval',
                         torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5)),
                         [(1, 4)])

    def test_deep_and_wide_stacks(self):
        layers = []
        previous = 4
        for _ in range(6):
            layers += [torch.nn.Linear(previous, 12), torch.nn.Tanh()]
            previous = 12
        layers += [torch.nn.Linear(previous, 2)]
        self.check_layer('deep_x6', torch.nn.Sequential(*layers), [(1, 4)])
        self.check_layer('wide_256',
                         torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.Tanh(),
                                             torch.nn.Linear(256, 2)),
                         [(1, 4)])

    def test_residual_skip_connection(self):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.activation = torch.nn.Tanh()

            def forward(self, x):
                return x + self.activation(self.linear(x))

        self.check_layer('residual', Residual(), [(1, 4)])

    def test_multi_input_concat(self):
        class TwoInputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 4)

            def forward(self, a, b):
                return self.linear(torch.cat([a, b], dim=1))

        self.check_layer('multi_input', TwoInputs(), [(1, 2), (1, 4)])

    def test_conv_is_rejected_with_a_clear_message(self):
        """Convolution needs a 3-D sliding window and must fail loudly."""
        module = torch.nn.Sequential(torch.nn.Conv1d(1, 4, 3, padding=1), torch.nn.Flatten())
        with self.assertRaises(Exception) as ctx:
            self.check_layer('conv1d', module, [(1, 1, 4)])
        self.assertIn("Operation 'Conv' not implemented", str(ctx.exception))


class TestEmbeddedInModel(unittest.TestCase):
    """A converted cell must be usable inside do_mpc.model.Model."""

    @unittest.skipUnless(TORCH_INSTALLED, 'torch is not installed')
    def test_lstm_cell_becomes_a_do_mpc_model(self):
        torch.manual_seed(0)
        cell = torch.nn.LSTMCell(N_IN, N_HID)
        tmp = tempfile.mkdtemp(prefix='_onnx_tmp_', dir=os.getcwd())
        try:
            path = os.path.join(tmp, 'cell.onnx')
            torch.onnx.export(cell, (torch.zeros(1, N_IN),
                                     (torch.zeros(1, N_HID), torch.zeros(1, N_HID))),
                              path, input_names=['x', 'h', 'c'],
                              dynamo=False, opset_version=OPSET)
            conv = ONNXConversion(onnx.load(path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        x_row = ca.SX.sym('x', 1, N_IN)
        h_row = ca.SX.sym('h', 1, N_HID)
        c_row = ca.SX.sym('c', 1, N_HID)
        conv.convert(x=x_row, h=h_row, c=c_row)
        h_out, c_out = conv.output_layers

        model = do_mpc.model.Model('discrete', 'SX')
        s_p = model.set_variable('_x', 'p', shape=(N_IN, 1))
        s_h = model.set_variable('_x', 'h', shape=(N_HID, 1))
        s_c = model.set_variable('_x', 'c', shape=(N_HID, 1))
        model.set_variable('_u', 'u', shape=(0, 1))

        # ONNX uses row vectors, do-mpc uses column vectors -> transpose at the boundary
        h_next = conv[h_out].T
        c_next = conv[c_out].T
        for sym, target in ((x_row, s_p.T), (h_row, s_h.T), (c_row, s_c.T)):
            h_next = ca.substitute(h_next, sym, target)
            c_next = ca.substitute(c_next, sym, target)

        model.set_rhs('p', h_next[:N_IN])
        model.set_rhs('h', h_next)
        model.set_rhs('c', c_next)
        model.setup()

        self.assertEqual(model.n_x, N_IN + 2 * N_HID)
        self.assertTrue(model.flags['setup'])

        # The model's own rhs function must reproduce PyTorch exactly.
        worst = 0.0
        rng = np.random.RandomState(7)
        for _ in range(100):
            pv = rng.randn(N_IN, 1).astype(np.float32)
            hv = rng.randn(N_HID, 1).astype(np.float32)
            cv = rng.randn(N_HID, 1).astype(np.float32)
            with torch.no_grad():
                ht, ct = cell(torch.tensor(pv).T, (torch.tensor(hv).T, torch.tensor(cv).T))
            ht_col, ct_col = ht.numpy().T, ct.numpy().T          # (N_HID, 1) each
            rhs = dm(model._rhs_fun(np.vstack([pv, hv, cv]), np.zeros((0, 1)),
                                    np.zeros((model.n_z, 1)), np.zeros((model.n_tvp, 1)),
                                    np.zeros((model.n_p, 1)), np.zeros((model.n_w, 1))))
            # state order is [p (N_IN), h (N_HID), c (N_HID)] and p was defined as
            # the first N_IN entries of h_next
            worst = max(worst,
                        float(np.abs(rhs[:N_IN] - ht_col[:N_IN]).max()),
                        float(np.abs(rhs[N_IN:N_IN + N_HID] - ht_col).max()),
                        float(np.abs(rhs[N_IN + N_HID:] - ct_col).max()))
        self.assertLess(worst, 1e-5, 'max deviation %.2e' % worst)
        print('Model._rhs_fun vs PyTorch LSTMCell: max diff = %.2e' % worst)


class TestRankTracking(unittest.TestCase):
    """ONNX rank tracking via ``onnx.shape_inference``.

    CasADi is strictly two-dimensional, so ``_to_casadi_shape`` drops leading
    singleton dimensions. Axis-referencing operators then need to know how many
    were dropped, otherwise a **positive** axis silently addresses the wrong
    dimension. ``ONNXConversion._collect_ranks`` records the rank of every tensor
    and ``convert`` publishes the offset to the operator before dispatch.

    All graphs here are built with ``onnx.helper``, so no torch is required.
    """

    VALUE = np.arange(6, dtype=np.float64).reshape(2, 3)

    def _run(self, nodes, output_shape, initializers=None):
        model = make_model(nodes, [tensor_info('x', (2, 3))],
                           [tensor_info('y', output_shape)],
                           initializers=initializers)
        conv = ONNXConversion(model)
        conv.convert(x=self.VALUE)
        return conv, np.asarray(conv['y'])

    @staticmethod
    def _axes(name, values):
        return numpy_helper.from_array(np.array(values, dtype=np.int64), name)

    def test_ranks_are_collected_for_intermediate_values(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('ReduceMean', ['u3'], ['y'], axes=[1])]
        conv, _ = self._run(nodes, (1, 3), [self._axes('unsq', [0])])
        self.assertEqual(conv.ranks.get('x'), 2)
        self.assertEqual(conv.ranks.get('u3'), 3,
                         'shape inference did not record the rank-3 intermediate')
        self.assertEqual(conv._axis_offset('u3'), 1)
        self.assertEqual(conv._axis_offset('x'), 0)

    def test_positive_axis_is_remapped_through_the_dropped_dimension(self):
        """The bug this whole mechanism exists for."""
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('ReduceMean', ['u3'], ['y'], axes=[1])]
        _, got = self._run(nodes, (1, 3), [self._axes('unsq', [0])])
        expected = self.VALUE.reshape(1, 2, 3).mean(axis=1)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected)

    def test_negative_axis_is_unaffected_by_rank_reduction(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('ReduceMean', ['u3'], ['y'], axes=[-1])]
        _, got = self._run(nodes, (2, 1), [self._axes('unsq', [0])])
        np.testing.assert_allclose(got.ravel(), self.VALUE.mean(axis=-1))

    def test_softmax_axis_after_rank_reduction(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('Softmax', ['u3'], ['y'], axis=1)]
        _, got = self._run(nodes, (2, 3), [self._axes('unsq', [0])])
        reference = self.VALUE.reshape(1, 2, 3)
        shifted = np.exp(reference - reference.max(axis=1, keepdims=True))
        expected = (shifted / shifted.sum(axis=1, keepdims=True)).reshape(2, 3)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_transpose_perm_keeping_dropped_dims_dropped(self):
        """perm=[0,2,1] on [1,B,C] is a plain 2-D transpose once axis 0 is dropped."""
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('Transpose', ['u3'], ['y'], perm=[0, 2, 1])]
        _, got = self._run(nodes, (3, 2), [self._axes('unsq', [0])])
        np.testing.assert_allclose(got, self.VALUE.T)

    def test_transpose_perm_identity_on_rank3(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('Transpose', ['u3'], ['y'], perm=[0, 1, 2])]
        _, got = self._run(nodes, (2, 3), [self._axes('unsq', [0])])
        np.testing.assert_allclose(got, self.VALUE)

    def test_transpose_perm_moving_a_dropped_dim_raises(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('Transpose', ['u3'], ['y'], perm=[2, 1, 0])]
        with self.assertRaises(Exception) as ctx:
            self._run(nodes, (3, 2), [self._axes('unsq', [0])])
        self.assertIn('dropped', str(ctx.exception))

    def test_rank2_graphs_are_unchanged(self):
        """Offset 0 must reproduce the plain behaviour exactly."""
        nodes = [helper.make_node('Softmax', ['x'], ['y'], axis=1)]
        _, got = self._run(nodes, (2, 3))
        shifted = np.exp(self.VALUE - self.VALUE.max(axis=1, keepdims=True))
        np.testing.assert_allclose(got, shifted / shifted.sum(axis=1, keepdims=True),
                                   rtol=1e-12)

    def test_axis_out_of_range_for_the_known_rank_raises(self):
        nodes = [helper.make_node('Softmax', ['x'], ['y'], axis=2)]
        with self.assertRaises(Exception) as ctx:
            self._run(nodes, (2, 3))
        self.assertIn('out of range', str(ctx.exception))

    def test_missing_rank_information_degrades_to_offset_zero(self):
        """With no offset recorded, behaviour must fall back safely."""
        ops = ONNXOperations()
        self.assertEqual(ops._resolve_axis(-1), 1)
        self.assertEqual(ops._resolve_axis(1), 1)
        self.assertEqual(ops._resolve_axis(0), 0)
        self.assertEqual(ops._resolve_axis(-2), 0)
        with self.assertRaises(Exception):
            ops._resolve_axis(2)

    def test_unsqueeze_then_squeeze_roundtrip(self):
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq0'], ['u3']),
                 helper.make_node('Squeeze', ['u3', 'unsq1'], ['y'])]
        initializers = [self._axes('unsq0', [0]), self._axes('unsq1', [0])]
        _, got = self._run(nodes, (2, 3), initializers)
        np.testing.assert_allclose(got, self.VALUE)

    def test_layer_normalization_group_spanning_both_axes_after_rank_reduction(self):
        """axis=1 on [1,B,C] normalises over B *and* C, i.e. the whole matrix."""
        scale = np.tile(np.array([1.5, 0.5, 2.0]), (2, 1)).astype(np.float32)
        bias = np.tile(np.array([0.25, -0.5, 0.75]), (2, 1)).astype(np.float32)
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('LayerNormalization', ['u3', 's', 'b'], ['y'],
                                  epsilon=1e-5, axis=1)]
        initializers = [self._axes('unsq', [0]),
                        numpy_helper.from_array(scale, 's'),
                        numpy_helper.from_array(bias, 'b')]
        _, got = self._run(nodes, (2, 3), initializers)
        flat = self.VALUE.reshape(1, 6)
        mean = flat.mean()
        variance = ((flat - mean) ** 2).mean()
        expected = ((flat - mean) / np.sqrt(variance + 1e-5)).reshape(2, 3) * scale + bias
        self.assertEqual(got.shape, (2, 3))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_layer_normalization_last_axis_after_rank_reduction_is_single_axis(self):
        """axis=-1 must stay a per-row reduction even on a rank-reduced tensor."""
        scale = np.array([1.5, 0.5, 2.0], dtype=np.float32)
        bias = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        nodes = [helper.make_node('Unsqueeze', ['x', 'unsq'], ['u3']),
                 helper.make_node('LayerNormalization', ['u3', 's', 'b'], ['y'],
                                  epsilon=1e-5, axis=-1)]
        initializers = [self._axes('unsq', [0]),
                        numpy_helper.from_array(scale, 's'),
                        numpy_helper.from_array(bias, 'b')]
        _, got = self._run(nodes, (2, 3), initializers)
        mean = self.VALUE.mean(axis=1, keepdims=True)
        variance = ((self.VALUE - mean) ** 2).mean(axis=1, keepdims=True)
        expected = (self.VALUE - mean) / np.sqrt(variance + 1e-5) * scale + bias
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)



if __name__ == '__main__':
    unittest.main()
