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
"""Use a trained LSTM as the plant model inside do-mpc  (manual weight extraction).

The key modelling idea
----------------------
MPC rolls the model forward over the prediction horizon, so what it needs is the
LSTM **cell** recurrence, not a sequence-in/sequence-out LSTM.  Expose the hidden
state as do-mpc states and the rollout happens automatically:

        x_k = [ p_k ; h_k ; c_k ]            <- _x  (1 observed + 2*n_hid)
        u_k =   u_k                          <- _u
        h_{k+1}, c_{k+1} = LSTMCell(u_k, (h_k, c_k))
        p_{k+1}          = head(h_{k+1})

    model.set_rhs('p', p_next); model.set_rhs('h', h_next); model.set_rhs('c', c_next)

do-mpc then instantiates that recurrence once per horizon step, which is exactly
the temporal unrolling of the LSTM.  Because h and c are ordinary states, MHE can
estimate them too, and robust multi-stage MPC works unchanged.

The plant here is a mass-spring-damper of which the surrogate observes ONLY the
position -- velocity must be inferred from history.  That is precisely the case
where a recurrent surrogate earns its keep over a static feed-forward network.

Requires: torch (pip install "do-mpc[full]").   Run from this directory.
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

# ---------------------------------------------------------------- plant setup
K_SPRING, C_DAMP, MASS, DT = 10.0, 2.0, 0.1, 0.1
N_HID    = 16
SETPOINT = 0.005
P_BOUND  = 0.01        # |position| bound
U_BOUND  = 0.1         # |force|    bound


def true_step(x, u):
    """Forward-Euler one-step map of the real plant (used to make training data)."""
    p, v = np.ravel(x)
    return np.array([[p + DT * v], [v + DT * (-K_SPRING * p - C_DAMP * v + u) / MASS]])


class Surrogate(torch.nn.Module):
    """LSTMCell + linear read-out, rolled out over a sequence.

    Defined at module scope (not inside ``train_surrogate``) so that tests can
    instantiate it with deterministic weights and exercise the conversion without
    having to train anything.
    """

    def __init__(self, n_in=2, n_hid=N_HID, n_out=1):
        super().__init__()
        self.cell = torch.nn.LSTMCell(n_in, n_hid)
        self.head = torch.nn.Linear(n_hid, n_out)

    def forward(self, seq):                 # seq: (batch, T, n_in), normalized
        batch = seq.shape[0]
        h = seq.new_zeros(batch, self.cell.hidden_size)
        c = seq.new_zeros(batch, self.cell.hidden_size)
        outs = []
        for t in range(seq.shape[1]):
            h, c = self.cell(seq[:, t, :], (h, c))
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


# ------------------------------------------------------------ 1. train the LSTM
def train_surrogate(n_traj=60, traj_len=60, n_epochs=1200, seed=0):
    """Train  p_{k+1} = f(p_k, u_k, h_k, c_k)  on rolled-out plant trajectories.

    Data MUST be collected as sequences with the hidden state carried along,
    because that is how MPC will use the network.  Standardizing is not
    optional here: see the FFN example (../ann_surrogate_model) for measured
    evidence of what happens without it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_in, seq_out = [], []
    for _ in range(n_traj):
        x = np.array([[np.random.uniform(-P_BOUND, P_BOUND)],
                      [np.random.uniform(-0.02, 0.02)]])
        ins, outs = [], []
        for _ in range(traj_len):
            u = np.random.uniform(-U_BOUND, U_BOUND)
            ins.append([x[0, 0], u])
            x = true_step(x, u)
            outs.append([x[0, 0]])
        seq_in.append(ins)
        seq_out.append(outs)
    seq_in = np.asarray(seq_in, dtype=np.float32)      # (n_traj, T, 2)
    seq_out = np.asarray(seq_out, dtype=np.float32)    # (n_traj, T, 1)

    in_mu, in_sd = seq_in.reshape(-1, 2).mean(0), seq_in.reshape(-1, 2).std(0)
    out_mu, out_sd = float(seq_out.mean()), float(seq_out.std())

    a = torch.tensor((seq_in - in_mu) / in_sd)
    b = torch.tensor((seq_out - out_mu) / out_sd)

    net = Surrogate()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(a), b)
        loss.backward()
        opt.step()

    stats = {'in_mu': in_mu, 'in_sd': in_sd, 'out_mu': out_mu, 'out_sd': out_sd,
             'rmse': float(((net(a).detach() - b) ** 2).mean().sqrt()) * out_sd}
    return net, stats


# ------------------------------------------- 2. one LSTM cell step -> CasADi
def lstm_cell_casadi(x_sym, h_sym, c_sym, cell):
    """Translate one ``torch.nn.LSTMCell`` step into a CasADi expression.

    PyTorch packs the gate weights row-block-wise in the order i, f, g, o.
    (The ONNX ``LSTM`` operator uses i, o, f, c instead -- if you read weights
    out of an ONNX graph you must reorder them.)

    NOTE: ``ca.vertsplit(v, n)`` treats ``n`` as the ROWS PER BLOCK, not the
    number of blocks.  Explicit slicing is used below to avoid that trap.
    """
    w_ih = cell.weight_ih.detach().numpy()                       # (4H, n_in)
    w_hh = cell.weight_hh.detach().numpy()                       # (4H, H)
    b_ih = cell.bias_ih.detach().numpy().reshape(-1, 1)
    b_hh = cell.bias_hh.detach().numpy().reshape(-1, 1)

    gates = ca.mtimes(w_ih, x_sym) + b_ih + ca.mtimes(w_hh, h_sym) + b_hh   # (4H,1)
    H = gates.shape[0] // 4
    g_i, g_f, g_g, g_o = gates[0:H], gates[H:2*H], gates[2*H:3*H], gates[3*H:4*H]

    sigmoid = lambda z: 1 / (1 + ca.exp(-z))          # noqa: E731
    c_new = sigmoid(g_f) * c_sym + sigmoid(g_i) * ca.tanh(g_g)
    h_new = sigmoid(g_o) * ca.tanh(c_new)
    return h_new, c_new


def build_lstm_model(net, stats, n_hid=None):
    """Wrap the trained cell in a discrete ``do_mpc.model.Model``.

    Args:
        net: Trained :py:class:`Surrogate` instance.
        stats: Dict with ``in_mu``, ``in_sd``, ``out_mu``, ``out_sd`` as returned
            by :py:func:`train_surrogate`.
        n_hid: Hidden size. Defaults to the module constant ``N_HID``; pass a
            smaller value to keep tests fast.
    """
    n_hid = N_HID if n_hid is None else int(n_hid)
    model = do_mpc.model.Model(model_type='discrete', symvar_type='SX')
    s_p = model.set_variable('_x', 'p', shape=(1, 1))
    s_h = model.set_variable('_x', 'h', shape=(n_hid, 1))
    s_c = model.set_variable('_x', 'c', shape=(n_hid, 1))
    s_u = model.set_variable('_u', 'u', shape=(1, 1))

    # fold the training-time standardization into the symbolic expression
    raw = ca.vertcat(s_p, s_u)
    norm = (raw - ca.SX(stats['in_mu'].reshape(2, 1))) / ca.SX(stats['in_sd'].reshape(2, 1))

    h_next, c_next = lstm_cell_casadi(norm, s_h, s_c, net.cell)
    p_next = ca.mtimes(net.head.weight.detach().numpy(), h_next) \
        + net.head.bias.detach().numpy().reshape(1, 1)
    p_next = p_next * stats['out_sd'] + stats['out_mu']

    # CasADi >= 3.8: build validation Functions BEFORE model.setup().  Afterwards
    # the leaf symbols are recreated internally and ca.Function(...) raises
    # "variables ... are free".  model._rhs_fun is the post-setup alternative.
    f_pre = ca.Function('f_pre', [s_p, s_h, s_c, s_u], [p_next, h_next, c_next])

    model.set_rhs('p', p_next)
    model.set_rhs('h', h_next)
    model.set_rhs('c', c_next)
    model.setup()
    return model, f_pre


# --------------------------------------------------------------- 3. controller
def build_mpc(model, n_horizon=None):
    """Configure the MPC. ``n_horizon`` defaults to 15; pass less in tests."""
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = DT
    mpc.settings.n_horizon = 15 if n_horizon is None else int(n_horizon)
    mpc.settings.supress_ipopt_output()

    err = (SETPOINT - model.x['p']) ** 2
    mpc.set_objective(lterm=err, mterm=err)
    mpc.set_rterm(u=1e-2)

    mpc.bounds['lower', '_x', 'p'] = -P_BOUND
    mpc.bounds['upper', '_x', 'p'] = P_BOUND
    # h and c are internal network states: bound them generously so IPOPT stays
    # well scaled, but never so tight that the recurrence becomes infeasible.
    mpc.bounds['lower', '_x', 'h'] = -10.0
    mpc.bounds['upper', '_x', 'h'] = 10.0
    mpc.bounds['lower', '_x', 'c'] = -30.0
    mpc.bounds['upper', '_x', 'c'] = 30.0
    mpc.bounds['lower', '_u', 'u'] = -U_BOUND
    mpc.bounds['upper', '_u', 'u'] = U_BOUND
    mpc.setup()
    return mpc


def dm(v):
    """CasADi DM -> numpy (avoids the casadi>=3.8 numpy-legacy FutureWarning)."""
    return np.array(ca.DM(v).full())


def validate(net, stats, f_pre, model, n_pts=500):
    print('[validate] CasADi expression vs PyTorch LSTMCell')
    wp = wh = wc = 0.0
    for _ in range(n_pts):
        pv = np.random.uniform(-P_BOUND, P_BOUND, size=(1, 1))
        uv = np.random.uniform(-U_BOUND, U_BOUND, size=(1, 1))
        hv = np.random.randn(N_HID, 1).astype(np.float32)
        cv = np.random.randn(N_HID, 1).astype(np.float32)
        norm_in = (np.vstack([pv, uv]).astype(np.float32)
                   - stats['in_mu'].reshape(2, 1)) / stats['in_sd'].reshape(2, 1)
        with torch.no_grad():
            ht, ct = net.cell(torch.tensor(norm_in).T,
                              (torch.tensor(hv).T, torch.tensor(cv).T))
            # head works in normalized space; de-normalize to compare with CasADi
            pt = float(net.head(ht).numpy().ravel()[0]) * stats['out_sd'] + stats['out_mu']
        pc, hc, cc = f_pre(pv, hv, cv, uv)
        wp = max(wp, abs(float(np.array(pc).ravel()[0]) - pt))
        wh = max(wh, np.abs(dm(hc) - ht.numpy().T).max())
        wc = max(wc, np.abs(dm(cc) - ct.numpy().T).max())
    print('           max|p diff| = %.2e   max|h diff| = %.2e   max|c diff| = %.2e'
          % (wp, wh, wc))

    print('[validate] do-mpc internal rhs_fun vs my expression')
    xt = np.array([[0.006]] + [[0.0]] * (2 * N_HID))
    ut = np.array([[0.03]])
    rhs = dm(model._rhs_fun(xt, ut, np.zeros((model.n_z, 1)), np.zeros((model.n_tvp, 1)),
                            np.zeros((model.n_p, 1)), np.zeros((model.n_w, 1))))
    # f_pre declares three outputs, so calling it returns a tuple of DM
    mine = np.vstack([dm(v) for v in f_pre(xt[0:1], xt[1:1 + N_HID], xt[1 + N_HID:2 * N_HID + 1], ut)])
    print('           max abs diff = %.2e   (0 => MPC integrates exactly this LSTM)'
          % np.abs(rhs - mine).max())


def main():
    print('=' * 74)
    print('LSTM surrogate as the do-mpc plant model')
    print('=' * 74)

    net, stats = train_surrogate()
    print('[1] LSTM surrogate trained: position RMSE = %.3e (%.2f%% of signal std)'
          % (stats['rmse'], 100 * stats['rmse'] / stats['out_sd']))

    model, f_pre = build_lstm_model(net, stats)
    print('[2] do_mpc Model built: n_x = %d  (1 observed + %d h + %d c), n_u = %d'
          % (model.n_x, N_HID, N_HID, model.n_u))

    validate(net, stats, f_pre, model)

    mpc = build_mpc(model)
    print('[3] MPC on the LSTM model: opt_x = %d variables, opt_p = %d parameters'
          % (mpc.opt_x_num.cat.shape[0], mpc.opt_p_num.cat.shape[0]))

    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = DT
    simulator.setup()

    x0 = np.array([[0.006]] + [[0.0]] * (2 * N_HID))
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()
    simulator.set_initial_guess()
    estimator = do_mpc.estimator.StateFeedback(model)

    timer = Timer()
    x = x0.copy()
    rows = []
    print('\n[4] closed loop (setpoint %.4f)' % SETPOINT)
    for _ in range(40):
        timer.tic()
        u0 = mpc.make_step(x)
        timer.toc()
        x = estimator.make_step(simulator.make_step(u0))
        rows.append([x[0, 0], u0[0, 0]])
    timer.info()

    r = np.array(rows)
    print('     step         p            u')
    for i in (0, 4, 9, 19, 39):
        print('     %4d  %10.6f  %9.5f' % (i, r[i, 0], r[i, 1]))
    print('\n[5] results')
    print('     final position   = %.6f   (setpoint %.4f)' % (r[-1, 0], SETPOINT))
    print('     tracking error   = %.2e' % abs(r[-1, 0] - SETPOINT))
    print('     steady-state u   = %.5f   (analytic k*sp = %.5f)'
          % (r[-1, 1], K_SPRING * SETPOINT))
    print('     max|u|           = %.5f   (bound %.2f)' % (np.abs(r[:, 1]).max(), U_BOUND))
    print('     p within bounds  : %s' % bool(np.all(np.abs(r[:, 0]) <= P_BOUND + 1e-9)))


if __name__ == '__main__':
    main()
