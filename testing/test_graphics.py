#
#   This file is part of do-mpc
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.

"""Regression tests for do_mpc.graphics.

The graphics module is backend-agnostic, so every test runs headless against
matplotlib's ``Agg`` backend. No window is opened and no file is written.

What is covered:

* ``Graphics`` construction from live module data and from a reloaded pickle
  (the post-processing path, which is how results are usually plotted);
* ``add_line`` for every variable type, plus its argument validation;
* the ``result_lines`` / ``pred_lines`` power-indexed attributes;
* ``plot_results`` / ``plot_predictions`` including the ``t_ind`` windowing and
  the ``store_full_solution`` precondition;
* ``reset_axes`` / ``reset_prop_cycle`` / ``clear``;
* ``default_plot`` including the subset lists and its name validation.

The assertions check the *state of the matplotlib objects* (line data, number of
lines, axes limits) rather than pixels, which keeps the tests fast, headless and
independent of the matplotlib version's rendering.
"""

import copy
import os
import sys
import unittest

import numpy as np

# Must happen before pyplot is imported anywhere in the process.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from importlib import reload

do_mpc_path = '../'
if do_mpc_path not in sys.path:
    sys.path.append(do_mpc_path)

import casadi as ca
import do_mpc

N_STEPS = 6


def build_model():
    """Tiny discrete plant with states, an input and an auxiliary expression."""
    model = do_mpc.model.Model('discrete', 'SX')
    x = model.set_variable('_x', 'x', shape=(2, 1))
    u = model.set_variable('_u', 'u', shape=(1, 1))
    model.set_expression('x_sum', x[0] + x[1])
    A = np.array([[0.85, 0.10], [-0.25, 0.90]])
    B = np.array([[0.10], [0.05]])
    model.set_rhs('x', A @ x + B @ u)
    model.setup()
    return model


def build_mpc(model, store_full_solution, n_horizon=5):
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = 0.1
    mpc.settings.n_horizon = n_horizon
    mpc.settings.store_full_solution = store_full_solution
    mpc.settings.supress_ipopt_output()
    mpc.set_objective(lterm=ca.sumsqr(model.x['x']), mterm=ca.sumsqr(model.x['x']))
    mpc.set_rterm(u=0.05)
    mpc.bounds['lower', '_x', 'x'] = np.array([[-2.0], [-2.0]])
    mpc.bounds['upper', '_x', 'x'] = np.array([[2.0], [2.0]])
    mpc.bounds['lower', '_u', 'u'] = -1.0
    mpc.bounds['upper', '_u', 'u'] = 1.0
    mpc.setup()
    return mpc


def run_closed_loop(model, store_full_solution, n_steps=N_STEPS, x0=None):
    mpc = build_mpc(model, store_full_solution)
    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = 0.1
    simulator.setup()
    estimator = do_mpc.estimator.StateFeedback(model)

    x0 = np.array([[0.8], [-0.5]]) if x0 is None else x0
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()
    simulator.set_initial_guess()

    x = x0.copy()
    for _ in range(n_steps):
        u0 = mpc.make_step(x)
        x = estimator.make_step(simulator.make_step(u0))
    return mpc, simulator, estimator


class GraphicsTestCase(unittest.TestCase):
    """Shared fixture: one closed loop with predictions, one without."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.mpc_full, cls.sim_full, cls.est_full = run_closed_loop(cls.model, True)
        cls.mpc_lean, cls.sim_lean, _ = run_closed_loop(cls.model, False)

    def setUp(self):
        # Every test gets its own figure so nothing leaks between tests.
        plt.close('all')

    def tearDown(self):
        plt.close('all')


class TestGraphicsConstruction(GraphicsTestCase):

    def test_accepts_mpc_data(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        self.assertIs(graphics.data, self.mpc_full.data)

    def test_accepts_simulator_data(self):
        graphics = do_mpc.graphics.Graphics(self.sim_full.data)
        self.assertIs(graphics.data, self.sim_full.data)

    def test_accepts_estimator_data(self):
        do_mpc.graphics.Graphics(self.est_full.data)

    def test_works_on_data_reloaded_from_pickle(self):
        """The post-processing path: save, reload, plot."""
        do_mpc.data.save_results([self.mpc_full, self.sim_full], 'test_graphics_tmp',
                                 result_path='./results/', overwrite=True)
        try:
            loaded = do_mpc.data.load_results('./results/test_graphics_tmp.pkl')
            self.assertIn('mpc', loaded)
            self.assertIn('simulator', loaded)
            graphics = do_mpc.graphics.Graphics(loaded['mpc'])
            fig, ax = plt.subplots(2, sharex=True)
            graphics.add_line('_x', 'x', ax[0])
            graphics.add_line('_u', 'u', ax[1])
            graphics.plot_results()
            line = graphics.result_lines['_x', 'x'][0]
            self.assertGreater(len(line.get_xdata()), 0)
        finally:
            path = './results/test_graphics_tmp.pkl'
            if os.path.isfile(path):
                os.remove(path)


class TestAddLine(GraphicsTestCase):

    def test_add_line_for_each_variable_type(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(3)
        graphics.add_line('_x', 'x', ax[0])
        graphics.add_line('_u', 'u', ax[1])
        graphics.add_line('_aux', 'x_sum', ax[2])
        self.assertEqual(len(graphics.result_lines['_x', 'x']), 2)   # vector state
        self.assertEqual(len(graphics.result_lines['_u', 'u']), 1)
        self.assertEqual(len(graphics.result_lines['_aux', 'x_sum']), 1)

    def test_input_lines_use_steps_post_drawstyle(self):
        """_u is a zero-order-hold signal, so it must be drawn as steps."""
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_u', 'u', ax)
        self.assertEqual(graphics.result_lines['_u', 'u'][0].get_drawstyle(), 'steps-post')

    def test_pltkwargs_are_forwarded_to_matplotlib(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax, linewidth=4.5, color='red', alpha=0.5)
        for line in graphics.result_lines['_x', 'x']:
            self.assertEqual(line.get_linewidth(), 4.5)
            self.assertEqual(line.get_alpha(), 0.5)

    def test_lines_are_attached_to_the_given_axis(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(2)
        graphics.add_line('_x', 'x', ax[0])
        graphics.add_line('_u', 'u', ax[1])
        # store_full_solution is on, so each variable contributes a result line
        # AND a dashed prediction line per vector component.
        self.assertEqual(len(ax[0].get_lines()), 4)      # 2 states x (result + prediction)
        self.assertEqual(len(ax[1].get_lines()), 2)      # 1 input  x (result + prediction)
        self.assertIn(ax[0], graphics.ax_list)
        self.assertIn(ax[1], graphics.ax_list)

    def test_prediction_lines_created_only_when_full_solution_stored(self):
        fig, ax = plt.subplots(1)
        with_full = do_mpc.graphics.Graphics(self.mpc_full.data)
        with_full.add_line('_x', 'x', ax)
        self.assertGreater(len(with_full.pred_lines.master), 0)

        without = do_mpc.graphics.Graphics(self.mpc_lean.data)
        without.add_line('_x', 'x', ax)
        self.assertEqual(len(without.pred_lines.master), 0)

    def test_invalid_var_type_raises(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        with self.assertRaises(AssertionError):
            graphics.add_line('_nope', 'x', ax)

    def test_non_string_arguments_raise(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        with self.assertRaises(AssertionError):
            graphics.add_line(42, 'x', ax)
        with self.assertRaises(AssertionError):
            graphics.add_line('_x', 42, ax)

    def test_non_axes_object_raises(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        with self.assertRaises(AssertionError):
            graphics.add_line('_x', 'x', fig)          # Figure, not Axes


class TestPlotting(GraphicsTestCase):

    def test_plot_results_fills_the_line_data(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.plot_results()
        line = graphics.result_lines['_x', 'x'][0]
        # MPC data records one row per make_step call.
        self.assertEqual(len(line.get_xdata()), N_STEPS)
        # ravel both sides: the line data is (n_t, 1) while the query is (n_t,),
        # and assert_allclose would otherwise broadcast them into an n_t x n_t
        # all-pairs comparison.
        np.testing.assert_allclose(np.asarray(line.get_ydata()).ravel(),
                                   np.asarray(self.mpc_full.data['_x', 'x', 0]).ravel())

    def test_plot_results_t_ind_windows_the_data(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.plot_results(t_ind=2)
        line = graphics.result_lines['_x', 'x'][0]
        self.assertEqual(len(line.get_xdata()), 3)       # indices 0..2

    def test_plot_results_t_ind_validation(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        with self.assertRaises(AssertionError):
            graphics.plot_results(t_ind=1.5)             # not an int
        with self.assertRaises(AssertionError):
            graphics.plot_results(t_ind=10 ** 6)         # out of range

    def test_plot_predictions_requires_mpc_data(self):
        graphics = do_mpc.graphics.Graphics(self.sim_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        with self.assertRaises(AssertionError):
            graphics.plot_predictions()

    def test_plot_predictions_requires_store_full_solution(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_lean.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        with self.assertRaises(AssertionError) as ctx:
            graphics.plot_predictions()
        self.assertIn('Optimal trajectory is not stored', str(ctx.exception))

    def test_plot_predictions_sets_the_prediction_line_data(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.plot_predictions(t_ind=0)
        n_horizon = self.mpc_full.data.meta_data['n_horizon']
        for line in graphics.pred_lines.master:
            self.assertEqual(len(line.get_xdata()), n_horizon + 1)
            self.assertEqual(line.get_linestyle(), '--')     # predictions are dashed

    def test_reset_axes_restores_autoscale(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.plot_results()
        ax.set_xlim(-100, 100)
        graphics.reset_axes()
        self.assertNotEqual(tuple(ax.get_xlim()), (-100.0, 100.0))

    def test_clear_empties_the_line_data(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.plot_results()
        self.assertGreater(len(graphics.result_lines['_x', 'x'][0].get_xdata()), 0)
        graphics.clear()
        self.assertEqual(len(graphics.result_lines['_x', 'x'][0].get_xdata()), 0)

    def test_reset_prop_cycle_runs(self):
        graphics = do_mpc.graphics.Graphics(self.mpc_full.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        graphics.reset_prop_cycle()          # smoke test: must not raise

    def test_plot_results_on_empty_data_raises(self):
        """A fresh MPC that never ran has zero recorded elements."""
        model = build_model()
        mpc = build_mpc(model, False)
        graphics = do_mpc.graphics.Graphics(mpc.data)
        fig, ax = plt.subplots(1)
        graphics.add_line('_x', 'x', ax)
        with self.assertRaises(AssertionError) as ctx:
            graphics.plot_results()
        self.assertIn('out of range for recorded data with 0 elements', str(ctx.exception))


class TestDefaultPlot(GraphicsTestCase):

    def test_returns_figure_axes_and_graphics(self):
        fig, ax, graphics = do_mpc.graphics.default_plot(self.mpc_full.data)
        self.assertIsInstance(graphics, do_mpc.graphics.Graphics)
        self.assertIsInstance(ax[0], Axes)
        # n_plot counts variable NAMES, not vector components:
        # 1 state ('x') + 0 dae + 1 input ('u') + 1 aux ('x_sum') = 3 axes
        self.assertEqual(len(ax), 3)
        self.assertEqual([a.get_ylabel() for a in ax], ['x', 'u', 'x_sum'])
        self.assertEqual(ax[-1].get_xlabel(), 'time')

    def test_figsize_kwarg_is_forwarded(self):
        fig, ax, _ = do_mpc.graphics.default_plot(self.mpc_full.data, figsize=(3, 7))
        self.assertEqual(tuple(np.round(fig.get_size_inches(), 3)), (3.0, 7.0))

    def test_subset_lists_restrict_the_plot(self):
        fig, ax, graphics = do_mpc.graphics.default_plot(
            self.mpc_full.data, states_list=['x'], inputs_list=['u'], aux_list=[])
        self.assertEqual(len(ax), 2)                     # no aux axis requested
        self.assertEqual([a.get_ylabel() for a in ax], ['x', 'u'])
        # Structure.__getitem__ returns an empty list for an unregistered key
        # rather than raising, so check the registry directly.
        self.assertEqual(graphics.result_lines['_aux', 'x_sum'], [])
        # One registry entry per vector COMPONENT: 'x' has two, 'u' has one.
        self.assertEqual(graphics.result_lines.powerindex,
                         [('_x', 'x', 0), ('_x', 'x', 1), ('_u', 'u', 0)])

    def test_invalid_subset_names_raise(self):
        with self.assertRaises(AssertionError):
            do_mpc.graphics.default_plot(self.mpc_full.data, states_list=['does_not_exist'])
        with self.assertRaises(AssertionError):
            do_mpc.graphics.default_plot(self.mpc_full.data, inputs_list=['does_not_exist'])
        with self.assertRaises(AssertionError):
            do_mpc.graphics.default_plot(self.mpc_full.data, aux_list=['does_not_exist'])

    def test_works_on_simulator_data(self):
        fig, ax, graphics = do_mpc.graphics.default_plot(self.sim_full.data)
        self.assertGreater(len(ax), 0)


if __name__ == '__main__':
    unittest.main()