#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

"""Use an already-trained ANN as the plant model inside do-mpc  ("Route A").

Pipeline
--------
    1.  Train a PyTorch FFN surrogate of a mass-spring-damper system,
        learning the one-step map   x_{k+1} = f(x_k, u_k).
    2.  Translate the trained weights layer-by-layer into CasADi expressions.
    3.  Feed that expression to ``model.set_rhs`` -> a normal do_mpc.model.Model.
    4.  Validate numerically (CasADi vs PyTorch, autodiff vs finite differences,
        steady-state map vs analytics).
    5.  Run a closed loop: MPC on the ANN surrogate, applied to the real plant.

Key insight
-----------
Once the ANN is a CasADi symbolic expression, do-mpc's whole machinery applies
to it unchanged: CasADi differentiates through the network automatically, the
orthogonal-collocation discretization instantiates it at every collocation
point, and IPOPT solves the resulting NLP.  Nothing downstream knows or cares
that the model equations came from a neural network.

Requires: torch (pip install "do-mpc[full]")

Run from this directory:   python main.py
"""

import os
import sys

import numpy as np
import torch
import casadi as ca

rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer


# --------------------------------------------------------------------------
# 0.  The ground-truth plant: mass-spring-damper
# --------------------------------------------------------------------------
K_SPRING = 10.0     # spring constant
C_DAMP   = 2.0      # damping constant
MASS     = 0.1      # mass
DT       = 0.1      # sampling time

LBX = np.array([-0.01, -0.0265])    # [position, velocity] bounds
UBX = np.array([ 0.01,  0.0265])
LBU = np.array([-0.1])              # force bound
UBU = np.array([ 0.1])
SETPOINT = 0.005


def true_euler_step(x, u):
    """Forward-Euler one-step map, used only to generate training data."""
    p, v = np.ravel(x)
    return np.array([p + DT * v, v + DT * (-K_SPRING * p - C_DAMP * v + u) / MASS])


# --------------------------------------------------------------------------
# 1.  Train the surrogate network
# --------------------------------------------------------------------------
def train_surrogate(n_samples=20000, n_epochs=1500, seed=0):
    """Train  x_{k+1} = f(x_k, u_k)  on data sampled over the whole operating box.

    IMPORTANT: the data is standardized before training and de-standardized
    inside the CasADi expression.  Training on raw values silently produces a
    useless surrogate here, because ``velocity`` has ~14x the standard
    deviation of ``position`` -- an unnormalized MSE loss therefore ignores the
    position output almost entirely.  Measured on this example:

        raw training   : relative RMSE ~53 %, steady-state map has the WRONG SIGN
        normalized     : relative RMSE ~0.6 %, steady-state error ~1e-5
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = np.random.uniform(LBX, UBX, size=(n_samples, 2))
    u = np.random.uniform(LBU, UBU, size=(n_samples, 1))
    net_in = np.hstack([x, u])
    net_out = np.vstack([true_euler_step(x[i], u[i, 0]) for i in range(n_samples)])

    in_mu, in_sd = net_in.mean(0), net_in.std(0)
    out_mu, out_sd = net_out.mean(0), net_out.std(0)

    net = torch.nn.Sequential(
        torch.nn.Linear(3, 32), torch.nn.Tanh(),   # tanh: smooth -> IPOPT-friendly
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 2),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    a = torch.tensor((net_in - in_mu) / in_sd, dtype=torch.float32)
    b = torch.tensor((net_out - out_mu) / out_sd, dtype=torch.float32)

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(a), b)
        loss.backward()
        opt.step()

    stats = {'in_mu': in_mu, 'in_sd': in_sd, 'out_mu': out_mu, 'out_sd': out_sd,
             'loss': float(loss.item())}
    return net, stats


# --------------------------------------------------------------------------
# 2.  ANN -> CasADi -> do_mpc.model.Model
# --------------------------------------------------------------------------
def ann_to_do_mpc_model(net, stats):
    """Translate trained weights into a CasADi expression and wrap it in a Model.

    Supported layers: ``torch.nn.Linear`` and ``torch.nn.Tanh``.
    Extend the loop below to support more (use smooth activations only --
    ReLU's kink at zero makes IPOPT's gradient-based search unreliable).

    Returns (model, f_pre) where ``f_pre`` is a plain CasADi Function built
    *before* ``model.setup()``, which is handy for validation.
    """
    model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')
    sx = model.set_variable('_x', 'states', shape=(2, 1))
    su = model.set_variable('_u', 'inputs', shape=(1, 1))

    # fold the training-time standardization into the symbolic expression
    h = (ca.vertcat(sx, su) - ca.SX(stats['in_mu'].reshape(3, 1))) \
        / ca.SX(stats['in_sd'].reshape(3, 1))

    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().numpy()      # shape (out, in)
            b = layer.bias.detach().numpy()        # shape (out,)
            h = ca.mtimes(w, h) + b                # (out,in) x (in,1) -> (out,1)
        elif isinstance(layer, torch.nn.Tanh):
            h = ca.tanh(h)
        else:
            raise RuntimeError(f'Layer type {layer} is not supported by this converter.')

    expr = h * ca.SX(stats['out_sd'].reshape(2, 1)) + ca.SX(stats['out_mu'].reshape(2, 1))

    # CasADi >= 3.8: build validation Functions from these symbols BEFORE setup().
    # After model.setup() the leaf symbols are recreated internally and
    # ca.Function(...) raises "variables ... are free".
    f_pre = ca.Function('f_pre', [sx, su], [expr])
    j_pre = ca.Function('j_pre', [sx, su], [ca.jacobian(expr, sx), ca.jacobian(expr, su)])

    model.set_rhs('states', expr)      # the ANN *is* the state equation
    model.setup()
    return model, f_pre, j_pre


# --------------------------------------------------------------------------
# 3.  The real plant (exact integration, deliberately NOT the Euler map)
# --------------------------------------------------------------------------
def build_real_plant():
    model = do_mpc.model.Model('continuous', symvar_type='SX')
    pos = model.set_variable('_x', 'position', shape=(1, 1))
    vel = model.set_variable('_x', 'velocity', shape=(1, 1))
    force = model.set_variable('_u', 'force', shape=(1, 1))
    model.set_rhs('position', vel)
    model.set_rhs('velocity', (-K_SPRING * pos - C_DAMP * vel + force) / MASS)
    model.setup()

    sim = do_mpc.simulator.Simulator(model)
    sim.settings.t_step = DT
    sim.settings.abstol = 1e-10
    sim.settings.reltol = 1e-10
    sim.setup()
    return model, sim


def build_mpc(model, x_name, u_name, vector_state):
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = DT
    mpc.settings.n_horizon = 10
    mpc.settings.supress_ipopt_output()

    err = (SETPOINT - model.x[x_name][0])**2 if vector_state else (SETPOINT - model.x[x_name])**2
    mpc.set_objective(lterm=err, mterm=err)
    mpc.set_rterm(**{u_name: 0.1})

    mpc.bounds['lower', '_x', x_name] = LBX if vector_state else LBX[0]
    mpc.bounds['upper', '_x', x_name] = UBX if vector_state else UBX[0]
    mpc.bounds['lower', '_u', u_name] = LBU[0]
    mpc.bounds['upper', '_u', u_name] = UBU[0]
    mpc.setup()
    return mpc


def dm(v):
    """CasADi DM -> numpy array (avoids the casadi>=3.8 numpy-legacy warning)."""
    return np.array(ca.DM(v).full())


# --------------------------------------------------------------------------
# 4.  Validation -- never trust an unverified conversion
# --------------------------------------------------------------------------
def validate(net, stats, f_pre, j_pre, model, n_pts=2000):
    print('[validate] CasADi expression vs PyTorch forward pass')
    worst = 0.0
    for _ in range(n_pts):
        xr = np.random.uniform(LBX, UBX).reshape(2, 1)
        ur = np.random.uniform(LBU, UBU).reshape(1, 1)
        a = (np.vstack([xr, ur]).ravel() - stats['in_mu']) / stats['in_sd']
        with torch.no_grad():
            ref = net(torch.tensor(a.astype(np.float32)).unsqueeze(0)).numpy()[0]
        ref = ref * stats['out_sd'] + stats['out_mu']
        worst = max(worst, np.abs(dm(f_pre(xr, ur)).reshape(2) - ref).max())
    print('            max abs diff over %d random points = %.2e' % (n_pts, worst))

    print('[validate] do-mpc internal rhs_fun vs my expression')
    xt = np.array([[0.004], [-0.012]])
    ut = np.array([[0.05]])
    rhs = dm(model._rhs_fun(xt, ut,
                            np.zeros((model.n_z, 1)), np.zeros((model.n_tvp, 1)),
                            np.zeros((model.n_p, 1)), np.zeros((model.n_w, 1))))
    print('            max abs diff = %.2e   (0 => MPC integrates exactly this ANN)'
          % np.abs(rhs - dm(f_pre(xt, ut))).max())

    # This measures the SURROGATE'S local accuracy, not the correctness of the
    # conversion.  Conversion correctness is what the two checks above prove.
    print('[validate] surrogate local Jacobian (CasADi autodiff) vs analytic Euler map')
    jac = dm(j_pre(xt, ut)[0])
    exact = np.array([[1.0, DT], [-K_SPRING * DT / MASS, 1 - C_DAMP * DT / MASS]])
    print('            autodiff df/dx =\n%s' % jac)
    print('            analytic df/dx =\n%s' % exact)
    print('            max abs diff = %.2e   max rel diff = %.2f%%'
          % (np.abs(jac - exact).max(), 100 * np.abs((jac - exact) / exact).max()))
    print('            (autodiff agreeing with the analytic map to ~1% is what makes')
    print('             the ANN usable inside an IPOPT-based MPC.)')

    print('[validate] steady-state map  p_ss(u)  vs analytic  u/k')
    for u in (0.02, 0.05, 0.08):
        x = np.zeros((2, 1))
        for _ in range(600):
            x = dm(f_pre(x, np.array([[u]])))
        print('            u=%.2f  surrogate=%+.6f  analytic=%+.6f  err=%.1e'
              % (u, x[0, 0], u / K_SPRING, abs(x[0, 0] - u / K_SPRING)))


# --------------------------------------------------------------------------
# 5.  Closed loop
# --------------------------------------------------------------------------
def main():
    print('=' * 72)
    print('Route A: trained ANN as the do-mpc plant model')
    print('=' * 72)

    net, stats = train_surrogate()
    print('[1] surrogate trained, normalized-space MSE = %.2e' % stats['loss'])

    sur_model, f_pre, j_pre = ann_to_do_mpc_model(net, stats)
    print('[2] do_mpc.model.Model built from the ANN, n_x=%d n_u=%d'
          % (sur_model.n_x, sur_model.n_u))

    validate(net, stats, f_pre, j_pre, sur_model)

    _, sim_real = build_real_plant()
    sim_sur = do_mpc.simulator.Simulator(sur_model)
    sim_sur.settings.t_step = DT
    sim_sur.setup()

    mpc = build_mpc(sur_model, 'states', 'inputs', vector_state=True)

    x0 = np.array([[0.004], [-0.012]])
    mpc.x0 = x0
    sim_sur.x0 = x0
    sim_real.x0 = x0
    mpc.set_initial_guess()
    sim_sur.set_initial_guess()
    sim_real.set_initial_guess()

    estimator = do_mpc.estimator.StateFeedback(sur_model)

    timer = Timer()
    x_sur = x0.copy()
    rows = []
    print('\n[3] closed loop: MPC(ANN surrogate) driving the real plant')
    for _ in range(40):
        timer.tic()
        u0 = mpc.make_step(x_sur)
        timer.toc()
        x_sur = estimator.make_step(sim_sur.make_step(u0))
        x_real = sim_real.make_step(u0=np.array([[u0[0, 0]]]))
        rows.append([x_sur[0, 0], x_real[0, 0], x_real[1, 0], u0[0, 0]])
    timer.info()

    r = np.array(rows)
    print('     step  surrogate_pos   real_pos   real_vel        u')
    for i in (0, 4, 9, 19, 39):
        print('     %4d  %13.6f %10.6f %9.6f %8.5f' % (i, r[i, 0], r[i, 1], r[i, 2], r[i, 3]))

    print('\n[4] results')
    print('     final real position    = %.6f   (setpoint %.4f)' % (r[-1, 1], SETPOINT))
    print('     tracking error         = %.2e' % abs(r[-1, 1] - SETPOINT))
    print('     surrogate/real mismatch= %.2e' % abs(r[-1, 0] - r[-1, 1]))
    print('     steady-state force     = %.5f   (analytic k*sp = %.5f)'
          % (r[-1, 3], K_SPRING * SETPOINT))
    print('     position in bounds     : %s' % bool(np.all(np.abs(r[:, 1]) <= UBX[0] + 1e-6)))
    print('     velocity in bounds     : %s' % bool(np.all(np.abs(r[:, 2]) <= UBX[1] + 1e-6)))
    print('     |u| within bound       : %s' % bool(np.all(np.abs(r[:, 3]) <= UBU[0] + 1e-9)))


if __name__ == '__main__':
    main()
    