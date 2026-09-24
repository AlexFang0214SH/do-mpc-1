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
"""Route 2: get the same LSTM into do-mpc through ONNX.

The operators this needs -- Constant, Split, Transpose, Identity, plus correct
Unsqueeze / Squeeze -- and the two convert() fixes (empty-string optional inputs
and multi-output nodes) are implemented in ``do_mpc.sysid`` itself, so this
script uses the library directly and needs no local extension module.

This script does three things:

  1. Shows what the STOCK ``do_mpc.sysid.ONNXConversion`` can and cannot handle
     for recurrent modules (nn.RNNCell works, nn.LSTMCell does not).
  2. Converts nn.LSTMCell / nn.GRUCell / nn.RNNCell through the stock
     ``do_mpc.sysid.ONNXConversion`` and validates each against PyTorch.
  3. Builds two do_mpc models from the SAME weights -- one by manual weight
     extraction, one through ONNX -- and proves they are identical, all the way
     down to the control sequence MPC produces.
  4. Rebuilds the same model through ``do_mpc.sysid.ann_to_dompc_model``, the
     declarative wrapper that does all the transposing / substituting / width
     checking for you, and shows that switching LSTM -> GRU only changes the spec.

Requires: torch, onnx  (pip install "do-mpc[full]").  Run from this directory.
"""

import os
import sys

import numpy as np
import torch
import casadi as ca
import onnx

rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.sysid import ONNXConversion

N_IN, N_HID, DT = 2, 8, 0.1
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp')
os.makedirs(TMP, exist_ok=True)


def dm(v):
    return np.array(ca.DM(v).full())


def export(mod, args, names, stem):
    path = os.path.join(TMP, stem + '.onnx')
    torch.onnx.export(mod, args, path, input_names=names,
                      dynamo=False,             # <- required: the dynamo exporter
                      opset_version=17)         #    emits ops this tool lacks
    return onnx.load(path)


def report_ops(tag, onnx_model):
    """List which ops a graph needs and whether the stock converter has them."""
    ops = sorted({n.op_type for n in onnx_model.graph.node})
    have = do_mpc.sysid.ONNXOperations
    missing = [o for o in ops if not hasattr(have, o)]
    multi = sorted({n.op_type for n in onnx_model.graph.node if len(n.output) > 1})
    empty = sorted({n.op_type for n in onnx_model.graph.node if '' in list(n.input)})
    print('  %-9s %2d nodes  ops=%s' % (tag, len(onnx_model.graph.node), ops))
    print('            missing from ONNXOperations : %s' % (missing or 'NONE'))
    print('            multi-output nodes          : %s' % (multi or 'none'))
    print('            empty-string (optional) in  : %s' % (empty or 'none'))
    return missing, multi, empty


# --------------------------------------------------------------------------
def part1_stock_converter():
    print('=' * 74)
    print('[1] What the stock do-mpc converter can already do')
    print('=' * 74)
    torch.manual_seed(0)

    rnn = torch.nn.RNNCell(N_IN, N_HID, nonlinearity='tanh')
    m_rnn = export(rnn, (torch.zeros(1, N_IN), torch.zeros(1, N_HID)),
                   ['x', 'h'], 'rnn')
    report_ops('RNNCell', m_rnn)
    xs, hs = ca.SX.sym('x', 1, N_IN), ca.SX.sym('h', 1, N_HID)
    conv = ONNXConversion(m_rnn)
    conv.convert(x=xs, h=hs)
    print('            -> STOCK converter: SUCCESS, output shape %s'
          % (conv[conv.output_layers[0]].shape,))

    lstm = torch.nn.LSTMCell(N_IN, N_HID)
    m_lstm = export(lstm, (torch.zeros(1, N_IN),
                           (torch.zeros(1, N_HID), torch.zeros(1, N_HID))),
                    ['x', 'h', 'c'], 'lstm')
    report_ops('LSTMCell', m_lstm)
    try:
        conv = ONNXConversion(m_lstm)
        # Constant and Split are the two operators this graph needs on top of the
        # feed-forward set, and Split is multi-output -- both are supported.
        x_sym = ca.SX.sym('x', 1, N_IN)
        h_sym = ca.SX.sym('h', 1, N_HID)
        c_sym = ca.SX.sym('c', 1, N_HID)
        conv.convert(x=x_sym, h=h_sym, c=c_sym)
        print('            -> STOCK converter: SUCCESS, outputs = %s'
              % (conv.output_layers,))
    except Exception as exc:
        print('            -> STOCK converter FAILED: %s' % str(exc).splitlines()[0][:100])

    # The sequence module is far worse than the cell.
    seq = torch.nn.LSTM(N_IN, N_HID, batch_first=True)
    m_seq = export(seq, torch.zeros(1, 4, N_IN), ['x'], 'lstm_seq')
    print('  %-9s %2d nodes  (the SEQUENCE module, for contrast)'
          % ('LSTM', len(m_seq.graph.node)))
    report_ops('LSTM(seq)', m_seq)
    try:
        ONNXConversion(m_seq).convert(x=np.zeros((1, 4, N_IN)))
    except Exception as exc:
        print('            -> STOCK converter FAILED: %s' % str(exc).splitlines()[0][:100])
    print('\n  Conclusion: export the CELL, never the sequence module.  The cell has no')
    print('  ONNX counterpart so torch decomposes it into primitives; the sequence')
    print('  module becomes one fused 3-D LSTM op plus a shape-juggling sub-graph.')


# --------------------------------------------------------------------------
def part2_extended_converter():
    print('\n' + '=' * 74)
    print('[2] do_mpc.sysid.ONNXConversion on LSTMCell / GRUCell / RNNCell')
    print('=' * 74)
    torch.manual_seed(0)
    results = {}
    cases = [
        ('LSTMCell', torch.nn.LSTMCell(N_IN, N_HID),
         (torch.zeros(1, N_IN), (torch.zeros(1, N_HID), torch.zeros(1, N_HID))),
         ['x', 'h', 'c']),
        ('GRUCell', torch.nn.GRUCell(N_IN, N_HID),
         (torch.zeros(1, N_IN), torch.zeros(1, N_HID)), ['x', 'h']),
        ('RNNCell', torch.nn.RNNCell(N_IN, N_HID, nonlinearity='tanh'),
         (torch.zeros(1, N_IN), torch.zeros(1, N_HID)), ['x', 'h']),
    ]
    for tag, mod, args, names in cases:
        mm = export(mod, args, names, tag.lower())
        conv = ONNXConversion(mm)
        syms = {n: ca.SX.sym(n, 1, sh[-1]) for n, sh in conv.inputshape.items()}
        conv.convert(**syms)
        f = ca.Function('f', [syms[n] for n in names],
                        [conv[o] for o in conv.output_layers])

        worst = 0.0
        for _ in range(300):
            xv = np.random.randn(1, N_IN).astype(np.float32)
            hv = np.random.randn(1, N_HID).astype(np.float32)
            cv = np.random.randn(1, N_HID).astype(np.float32)
            with torch.no_grad():
                if tag == 'LSTMCell':
                    ht, ct = mod(torch.tensor(xv), (torch.tensor(hv), torch.tensor(cv)))
                    ref = [ht, ct]
                    got = f(xv, hv, cv)
                else:
                    ref = [mod(torch.tensor(xv), torch.tensor(hv))]
                    got = f(xv, hv)
            got = got if isinstance(got, (list, tuple)) else [got]
            for g, rr in zip(got, ref):
                worst = max(worst, np.abs(dm(g) - rr.numpy()).max())
        print('  %-9s ONNX outputs %-30s 300 pts: max|casadi-torch| = %.2e'
              % (tag, str(conv.output_layers)[:30], worst))
        if tag == 'LSTMCell':
            results['conv'] = conv
            results['cell'] = mod
    return results


# --------------------------------------------------------------------------
def cell_casadi(x, h, c, cell):
    """Manual translation, identical to ../lstm_surrogate_model/main.py."""
    w_ih = cell.weight_ih.detach().numpy()
    w_hh = cell.weight_hh.detach().numpy()
    b_ih = cell.bias_ih.detach().numpy().reshape(-1, 1)
    b_hh = cell.bias_hh.detach().numpy().reshape(-1, 1)
    g = ca.mtimes(w_ih, x) + b_ih + ca.mtimes(w_hh, h) + b_hh
    H = g.shape[0] // 4
    gi, gf, gg, go = g[0:H], g[H:2 * H], g[2 * H:3 * H], g[3 * H:4 * H]
    sig = lambda z: 1 / (1 + ca.exp(-z))          # noqa: E731
    cn = sig(gf) * c + sig(gi) * ca.tanh(gg)
    return sig(go) * ca.tanh(cn), cn


def _rhs(model, x, u):
    return dm(model._rhs_fun(x, u, np.zeros((model.n_z, 1)), np.zeros((model.n_tvp, 1)),
                             np.zeros((model.n_p, 1)), np.zeros((model.n_w, 1))))


def part3_equivalence(cell):
    print('\n' + '=' * 74)
    print('[3] Manual path vs ONNX path, same weights -> same MPC')
    print('=' * 74)
    head = torch.nn.Linear(N_HID, 1)
    mm = export(cell, (torch.zeros(1, N_IN), (torch.zeros(1, N_HID), torch.zeros(1, N_HID))),
                ['x', 'h', 'c'], 'lstm_eq')
    conv = ONNXConversion(mm)
    xs = ca.SX.sym('x', 1, N_IN)
    hs = ca.SX.sym('h', 1, N_HID)
    cs = ca.SX.sym('c', 1, N_HID)
    conv.convert(x=xs, h=hs, c=cs)
    out_h, out_c = conv.output_layers          # h_next first, then c_next

    def build(use_onnx):
        model = do_mpc.model.Model('discrete', 'SX')
        sp = model.set_variable('_x', 'p', shape=(1, 1))
        sh = model.set_variable('_x', 'h', shape=(N_HID, 1))
        sc = model.set_variable('_x', 'c', shape=(N_HID, 1))
        su = model.set_variable('_u', 'u', shape=(1, 1))
        if use_onnx:
            # ONNX/torch use ROW vectors (1,N); do-mpc uses COLUMN vectors (N,1).
            # Transpose at the boundary and substitute the model symbols in.
            hn = conv[out_h].T
            cn = conv[out_c].T
            for sym, target in ((xs, ca.vertcat(sp, su).T), (hs, sh.T), (cs, sc.T)):
                hn = ca.substitute(hn, sym, target)
                cn = ca.substitute(cn, sym, target)
        else:
            hn, cn = cell_casadi(ca.vertcat(sp, su), sh, sc, cell)
        pn = ca.mtimes(head.weight.detach().numpy(), hn) \
            + head.bias.detach().numpy().reshape(1, 1)
        model.set_rhs('p', pn)
        model.set_rhs('h', hn)
        model.set_rhs('c', cn)
        model.setup()
        return model

    m_manual, m_onnx = build(False), build(True)
    print('  manual model n_x = %d   onnx model n_x = %d' % (m_manual.n_x, m_onnx.n_x))

    worst = 0.0
    for _ in range(500):
        xv = np.array([[np.random.uniform(-0.01, 0.01)]])
        uv = np.array([[np.random.uniform(-0.1, 0.1)]])
        hv = np.random.randn(N_HID, 1)
        cv = np.random.randn(N_HID, 1)
        x = np.vstack([xv, hv, cv])
        worst = max(worst, np.abs(_rhs(m_manual, x, uv) - _rhs(m_onnx, x, uv)).max())
    print('  max|rhs_fun(manual) - rhs_fun(onnx)| over 500 pts = %.2e' % worst)

    def build_mpc(model):
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.t_step = DT
        mpc.settings.n_horizon = 10
        mpc.settings.supress_ipopt_output()
        e = (0.005 - model.x['p']) ** 2
        mpc.set_objective(lterm=e, mterm=e)
        mpc.set_rterm(u=1e-2)
        mpc.bounds['lower', '_x', 'p'] = -0.01
        mpc.bounds['upper', '_x', 'p'] = 0.01
        mpc.bounds['lower', '_x', 'h'] = -10
        mpc.bounds['upper', '_x', 'h'] = 10
        mpc.bounds['lower', '_x', 'c'] = -30
        mpc.bounds['upper', '_x', 'c'] = 30
        mpc.bounds['lower', '_u', 'u'] = -0.1
        mpc.bounds['upper', '_u', 'u'] = 0.1
        mpc.setup()
        return mpc

    x0 = np.array([[0.006]] + [[0.0]] * (2 * N_HID))
    seqs = []
    for model in (m_manual, m_onnx):
        mpc = build_mpc(model)
        mpc.x0 = x0
        mpc.set_initial_guess()
        x, acc = x0.copy(), []
        for _ in range(8):
            u = mpc.make_step(x)
            acc.append(float(u[0, 0]))
            x = _rhs(model, x, u)
        seqs.append(acc)
    print('  MPC u (manual): %s' % np.round(seqs[0], 8))
    print('  MPC u (onnx)  : %s' % np.round(seqs[1], 8))
    print('  max|u_manual - u_onnx| over 8 MPC steps = %.2e'
          % np.max(np.abs(np.array(seqs[0]) - np.array(seqs[1]))))
    print('  (weights here are RANDOM, not trained -- this compares the two')
    print('   conversion routes, it is not a meaningful controller.)')


def part4_declarative_builder():
    """The same LSTM, built through do_mpc.sysid.ann_to_dompc_model.

    part3 does the ONNX -> Model wiring by hand: transpose the row vectors,
    ca.substitute every placeholder, assign each output to a state. That is ~25
    lines of fiddly, easy-to-get-wrong code. The builder replaces all of it with
    a declarative spec, and validates the widths up front so a mismatch is
    reported as such instead of surfacing later as an opaque
    "Error in powerIndex slicing" from casadi.tools.structure.
    """
    print('\n' + '=' * 74)
    print('[4] do_mpc.sysid.ann_to_dompc_model -- the declarative route')
    print('=' * 74)
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(N_IN, N_HID)
    head = torch.nn.Linear(N_HID, 1)

    class Plant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell, self.head = cell, head

        def forward(self, xu, h, c):
            h_next, c_next = self.cell(xu, (h, c))
            return self.head(h_next), h_next, c_next

    model, converter = do_mpc.sysid.ann_to_dompc_model(
        Plant(),
        x_spec=[('p', 1), ('h', N_HID), ('c', N_HID)],
        u_spec=[('u', 1)],
        wiring=[('xu', [('x', 'p'), ('u', 'u')]),
                ('h', [('x', 'h')]),
                ('c', [('x', 'c')])],
        outputs=[(0, 'p'), (1, 'h'), (2, 'c')],
        sample_args=(torch.zeros(1, N_IN), torch.zeros(1, N_HID), torch.zeros(1, N_HID)),
        input_names=['xu', 'h', 'c'],
        verbose=True)
    print('  built model: n_x = %d (1 observed + %d h + %d c), n_u = %d'
          % (model.n_x, N_HID, N_HID, model.n_u))

    # The builder is architecture-agnostic: swapping the cell for a GRU only
    # changes the spec, never the conversion code.
    gru = torch.nn.GRUCell(N_IN, N_HID)

    class GRUPlant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell, self.head = gru, head

        def forward(self, xu, h):
            h_next = self.cell(xu, h)
            return self.head(h_next), h_next

    gru_model, _ = do_mpc.sysid.ann_to_dompc_model(
        GRUPlant(),
        x_spec=[('p', 1), ('h', N_HID)],
        u_spec=[('u', 1)],
        wiring=[('xu', [('x', 'p'), ('u', 'u')]), ('h', [('x', 'h')])],
        outputs=[(0, 'p'), (1, 'h')],
        sample_args=(torch.zeros(1, N_IN), torch.zeros(1, N_HID)),
        input_names=['xu', 'h'])
    print('  same builder, GRU instead of LSTM: n_x = %d (no cell state)'
          % gru_model.n_x)

    # A deliberately wrong spec is rejected up front with an actionable message.
    # Here the state 'p' is declared 3-wide, so the wiring sums to 3+1 = 4 while
    # the graph input 'xu' is only 2-wide. Without this check the mismatch would
    # surface much later as an opaque casadi.tools.structure error.
    try:
        do_mpc.sysid.ann_to_dompc_model(
            Plant(), x_spec=[('p', 3), ('h', N_HID), ('c', N_HID)], u_spec=[('u', 1)],
            wiring=[('xu', [('x', 'p'), ('u', 'u')]), ('h', [('x', 'h')]),
                    ('c', [('x', 'c')])],
            outputs=[(0, 'p'), (1, 'h'), (2, 'c')],
            sample_args=(torch.zeros(1, N_IN), torch.zeros(1, N_HID), torch.zeros(1, N_HID)),
            input_names=['xu', 'h', 'c'])
    except Exception as exc:
        print('  a wrong spec is caught up front:')
        print('    %s' % str(exc).splitlines()[0][:110])


def main():
    np.random.seed(0)
    part1_stock_converter()
    res = part2_extended_converter()
    part3_equivalence(res['cell'])
    part4_declarative_builder()
    print('\nDone. Temp ONNX files are in %s' % TMP)


if __name__ == '__main__':
    main()