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

"""Turn a trained ANN into a :py:class:`do_mpc.model.Model`.

:py:class:`do_mpc.sysid.ONNXConversion` translates an ONNX graph into CasADi
expressions, but wiring those expressions into a model is repetitive and
error-prone: the ONNX tensors are row vectors while do-mpc variables are column
vectors, a recurrent network's hidden state has to be promoted to a model state,
and any size mismatch otherwise surfaces much later as an opaque
``Error in powerIndex slicing`` from ``casadi.tools.structure``.

:py:func:`ann_to_dompc_model` does that wiring once, driven by a **declarative
spec**, so switching to a different network means editing the spec rather than
the code.

The four things a spec has to say -- none of which can be inferred from the
weights -- are:

1. ``x_spec`` / ``u_spec``: which do-mpc states and inputs exist, and their sizes.
   For a recurrent network the hidden state belongs here, so that MPC's rollout
   over the prediction horizon *is* the network's recurrence.
2. ``wiring``: which do-mpc variables feed each ONNX graph input, and in what
   concatenation order.
3. ``outputs``: which graph output drives which state.
4. ``source``: the network itself (an ONNX model, a path, or a torch module).

Example
-------
A feed-forward surrogate of :math:`x_{k+1} = f(x_k, u_k)`:

::

    model, conv = do_mpc.sysid.ann_to_dompc_model(
        net,
        x_spec  = [('x', 2)],
        u_spec  = [('u', 2)],
        wiring  = [('xu', [('x', 'x'), ('u', 'u')])],   # ONNX input 'xu' = [x; u]
        outputs = [(0, 'x')],                           # graph output 0 drives 'x'
        sample_args = (torch.zeros(1, 4),))

An LSTM surrogate, where the hidden state becomes part of ``_x``:

::

    model, conv = do_mpc.sysid.ann_to_dompc_model(
        lstm_plant,
        x_spec  = [('x', 2), ('h', 8), ('c', 8)],
        u_spec  = [('u', 1)],
        wiring  = [('xu', [('x', 'x'), ('u', 'u')]),
                   ('h',  [('x', 'h')]),
                   ('c',  [('x', 'c')])],
        outputs = [(0, 'x'), (1, 'h'), (2, 'c')],
        input_names = ['xu', 'h', 'c'])

Warning:
    Only ``model_type='discrete'`` is recommended for learned surrogates. With a
    ``'continuous'`` model the orthogonal collocation scheme instantiates the
    network expression at ``n_horizon * collocation_ni * (collocation_deg + 1)``
    points, which makes the NLP grow very quickly.
"""

import inspect
import io
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import casadi

import do_mpc
from ._onnxconversion import ONNXConversion

try:
    import onnx
    __ONNX_AVAILABLE__ = True
except ImportError:      # pragma: no cover - onnx is a declared optional extra
    __ONNX_AVAILABLE__ = False

try:
    import torch
    __TORCH_AVAILABLE__ = True
except ImportError:      # pragma: no cover - torch is a declared optional extra
    __TORCH_AVAILABLE__ = False


__all__ = ['ann_to_dompc_model', 'torch_module_to_onnx', 'validate_wiring',
           'set_constant_parameters', 'set_constant_tvp', 'resolve_output']


def torch_module_to_onnx(module: Any,
                         sample_args: Sequence[Any],
                         input_names: Optional[Sequence[str]] = None,
                         opset: int = 17,
                         path: Optional[str] = None):
    """Export a ``torch.nn.Module`` to an ``onnx.ModelProto``.

    The model is exported **in memory**; nothing is written to disk unless
    ``path`` is given.

    Note:
        ``dynamo=False`` is required. The newer ``torch.export``-based exporter
        emits operators that :py:class:`do_mpc.sysid.ONNXOperations` does not
        implement.

    Args:
        module: The network. It is switched to ``eval()`` mode first, so
            ``Dropout`` disappears and ``BatchNormalization`` uses its running
            statistics.
        sample_args: Example inputs, used only to trace the graph. One entry per
            network input, each of shape ``(1, n)``.
        input_names: Names for the graph inputs. These are the names the
            ``wiring`` spec refers to, so supplying them explicitly is strongly
            recommended.
        opset: ONNX opset version.
        path: Optional file to write the model to.

    Returns:
        onnx.ModelProto: The exported graph.

    Raises:
        Exception: if torch or onnx is not installed.
    """
    if not __TORCH_AVAILABLE__:
        raise Exception('torch_module_to_onnx requires torch. '
                        'Install it with: pip install "do-mpc[full]"')
    if not __ONNX_AVAILABLE__:
        raise Exception('torch_module_to_onnx requires onnx. '
                        'Install it with: pip install "do-mpc[full]"')

    module.eval()
    kwargs = dict(dynamo=False, opset_version=opset)
    if input_names is not None:
        kwargs['input_names'] = list(input_names)
    if path is not None:
        torch.onnx.export(module, tuple(sample_args), path, **kwargs)
        return onnx.load(path)

    # torch.onnx.export requires a destination, so use an in-memory buffer and
    # parse it back. Nothing touches the filesystem this way.
    buffer = io.BytesIO()
    torch.onnx.export(module, tuple(sample_args), buffer, **kwargs)
    return onnx.load_model_from_string(buffer.getvalue())


def _is_supported_source(source: Any) -> bool:
    """True if ``source`` is something :py:func:`_load_source` can handle."""
    if __ONNX_AVAILABLE__ and isinstance(source, onnx.ModelProto):
        return True
    if isinstance(source, str):
        return True
    return bool(__TORCH_AVAILABLE__ and isinstance(source, torch.nn.Module))


def _load_source(source: Any,
                 sample_args: Optional[Sequence[Any]],
                 input_names: Optional[Sequence[str]],
                 opset: int):
    """Normalise ``source`` to an ``onnx.ModelProto``."""
    if __ONNX_AVAILABLE__ and isinstance(source, onnx.ModelProto):
        return source
    if isinstance(source, str):
        if not __ONNX_AVAILABLE__:
            raise Exception('Loading %r requires onnx. '
                            'Install it with: pip install "do-mpc[full]"' % source)
        if not os.path.isfile(source):
            raise Exception('ONNX file not found: %r' % source)
        return onnx.load(source)
    if __TORCH_AVAILABLE__ and isinstance(source, torch.nn.Module):
        if sample_args is None:
            raise Exception('Exporting a torch module needs sample_args: one example '
                            'tensor per network input, each of shape (1, n).')
        return torch_module_to_onnx(source, sample_args, input_names, opset)
    raise Exception('source must be an onnx.ModelProto, a path to a .onnx file, or a '
                    'torch.nn.Module (the latter also needs sample_args). You have: {}'
                    .format(type(source).__name__))


def validate_wiring(x_spec, u_spec, wiring, outputs, p_spec=(), tvp_spec=(),
                    z_spec=(), algebraic=(), measurements=()) -> None:
    """Check a spec for internal consistency before anything is built.

    Args:
        x_spec: ``[(name, size), ...]`` model states.
        u_spec: ``[(name, size), ...]`` model inputs.
        wiring: ``[(onnx_input_name, [(kind, var_name), ...]), ...]``.
        outputs: ``[(onnx_output_key, state_name), ...]``.
        p_spec: ``[(name, size), ...]`` model parameters (``_p``).
        tvp_spec: ``[(name, size), ...]`` time-varying parameters (``_tvp``).
        z_spec: ``[(name, size), ...]`` algebraic states (``_z``), for DAE models.
        algebraic: ``[(key, z_name), ...]`` assigning graph outputs to the
            algebraic equations ``0 = g(...)``.
        measurements: ``[(source, name, meas_noise), ...]`` or
            ``[(source, name), ...]`` declaring measurement equations.

    Raises:
        Exception: with a message naming the offending entry.
    """
    for label, spec in (('x_spec', x_spec), ('u_spec', u_spec),
                        ('p_spec', p_spec), ('tvp_spec', tvp_spec),
                        ('z_spec', z_spec)):
        names = [name for name, _ in spec]
        if len(names) != len(set(names)):
            raise Exception('%s contains duplicate variable names: %s' % (label, names))
        for name, size in spec:
            if not isinstance(name, str) or not name:
                raise Exception('%s entries must be (name, size) with a non-empty '
                                'string name; got %r' % (label, (name, size)))
            if int(size) < 0:
                raise Exception('%s entry %r has a negative size' % (label, name))

    declared = ({('x', name) for name, _ in x_spec}
                | {('u', name) for name, _ in u_spec}
                | {('p', name) for name, _ in p_spec}
                | {('tvp', name) for name, _ in tvp_spec}
                | {('z', name) for name, _ in z_spec})
    seen_inputs = []
    for entry in wiring:
        if len(entry) != 2:
            raise Exception('wiring entries must be (onnx_input_name, [(kind, var), ...]); '
                            'got %r' % (entry,))
        onnx_name, parts = entry
        seen_inputs.append(onnx_name)
        if not parts:
            raise Exception('wiring entry %r concatenates nothing' % onnx_name)
        for part in parts:
            if len(part) != 2:
                raise Exception('wiring parts must be (kind, var_name) with kind in '
                                "('x', 'u', 'p', 'tvp', 'z'); got %r" % (part,))
            if part[0] not in ('x', 'u', 'p', 'tvp', 'z'):
                raise Exception("wiring kind %r is not one of 'x', 'u', 'p', 'tvp', 'z'. "
                                "Noise variables (_w, _v) are generated by do-mpc and "
                                "cannot be network inputs." % (part[0],))
            if tuple(part) not in declared:
                raise Exception('wiring entry %r references %r, which is not declared in '
                                'x_spec, u_spec, p_spec, tvp_spec or z_spec. Declared: %s'
                                % (onnx_name, tuple(part), sorted(declared)))
    if len(seen_inputs) != len(set(seen_inputs)):
        raise Exception('wiring lists the same ONNX input twice: %s' % seen_inputs)

    x_names = [name for name, _ in x_spec]
    z_names = [name for name, _ in z_spec]
    for entry in outputs:
        if len(entry) != 2:
            raise Exception('outputs entries must be (onnx_output_key, state_name); '
                            'got %r' % (entry,))
        key, state = entry
        if state not in x_names:
            raise Exception('outputs entry %r drives %r, which is not a state in x_spec. '
                            'States: %s' % (entry, state, x_names))
    driven = [state for _, state in outputs]
    missing = [name for name in x_names if name not in driven]
    if missing:
        raise Exception('every state needs a driving output, but %s would never be '
                        'updated. Add them to `outputs`.' % missing)
    if len(driven) != len(set(driven)):
        raise Exception('outputs drives the same state twice: %s' % driven)

    # Algebraic equations: do-mpc requires exactly n_z of them, one per _z.
    algebraic_targets = []
    for entry in algebraic:
        if len(entry) != 2:
            raise Exception('algebraic entries must be (onnx_output_key, z_name); '
                            'got %r' % (entry,))
        key, z_name = entry
        if z_name not in z_names:
            raise Exception('algebraic entry %r targets %r, which is not in z_spec. '
                            'Algebraic states: %s' % (entry, z_name, z_names))
        algebraic_targets.append(z_name)
    missing_alg = [name for name in z_names if name not in algebraic_targets]
    if missing_alg:
        raise Exception('every algebraic state needs exactly one equation, but %s has '
                        'none. do-mpc asserts n_z == number of algebraic equations.'
                        % missing_alg)
    if len(algebraic_targets) != len(set(algebraic_targets)):
        raise Exception('algebraic supplies more than one equation for the same state: %s'
                        % algebraic_targets)

    # Measurements
    measurement_names = []
    for entry in measurements:
        if len(entry) not in (2, 3):
            raise Exception('measurements entries must be (source, name) or '
                            '(source, name, meas_noise); got %r' % (entry,))
        if len(entry) == 3 and not isinstance(entry[2], bool):
            raise Exception('measurements entry %r: meas_noise must be a bool' % (entry,))
        measurement_names.append(entry[1])
    if len(measurement_names) != len(set(measurement_names)):
        raise Exception('measurements declares the same name twice: %s' % measurement_names)


def ann_to_dompc_model(source: Any,
                       x_spec: Sequence[Tuple[str, int]],
                       u_spec: Sequence[Tuple[str, int]],
                       wiring: Sequence[Tuple[str, Sequence[Tuple[str, str]]]],
                       outputs: Sequence[Tuple[Union[int, str], str]],
                       p_spec: Sequence[Tuple[str, int]] = (),
                       tvp_spec: Sequence[Tuple[str, int]] = (),
                       z_spec: Sequence[Tuple[str, int]] = (),
                       algebraic: Sequence[Tuple[Any, str]] = (),
                       measurements: Sequence[Tuple[Any, str]] = (),
                       sample_args: Optional[Sequence[Any]] = None,
                       input_names: Optional[Sequence[str]] = None,
                       opset: int = 17,
                       model_type: str = 'discrete',
                       symvar_type: str = 'SX',
                       verbose: bool = False
                       ) -> Tuple[do_mpc.model.Model, ONNXConversion]:
    r"""Build a :py:class:`do_mpc.model.Model` whose equations are a trained ANN.

    The network is converted to CasADi expressions with
    :py:class:`do_mpc.sysid.ONNXConversion` and then wired into the model
    according to the declarative spec. Everything downstream in do-mpc --
    automatic differentiation, discretization, constraints, IPOPT, MHE, robust
    multi-stage MPC -- then applies to it unchanged.

    Args:
        source: ``onnx.ModelProto``, a path to a ``.onnx`` file, or a
            ``torch.nn.Module`` (the latter also needs ``sample_args``).
        x_spec: ``[(name, size), ...]`` for the model states. Recurrent hidden
            states belong here.
        u_spec: ``[(name, size), ...]`` for the model inputs. May be empty.
        wiring: ``[(onnx_input_name, [(kind, var_name), ...]), ...]`` giving, for
            each graph input, which do-mpc variables to concatenate and in what
            order. ``kind`` is one of ``'x'``, ``'u'``, ``'p'``, ``'tvp'`` or
            ``'z'`` -- the last is what makes an implicit DAE
            :math:`0 = g(x, u, z)` expressible. The concatenation is turned into
            a row vector to match the ONNX convention. Noise variables (``_w``,
            ``_v``) are generated by do-mpc and cannot be network inputs.
        outputs: ``[(key, state_name), ...]`` assigning each graph output to the
            state it drives. ``key`` is the output name or its index in
            ``converter.output_layers``.
        p_spec: ``[(name, size), ...]`` for model parameters (``_p``). Use this for
            a network conditioned on a constant-but-uncertain quantity, e.g. an
            operating point or a physical constant. Parameters are constant over
            the whole prediction horizon and can be estimated by
            :py:class:`do_mpc.estimator.MHE` or treated as uncertainties by robust
            multi-stage MPC.
        tvp_spec: ``[(name, size), ...]`` for time-varying parameters (``_tvp``).
            Use this for a network conditioned on a **known future sequence**, e.g.
            a reference trajectory, a weather forecast or an upstream disturbance
            prediction. Unlike ``_p``, a ``_tvp`` takes a different value at every
            step of the prediction horizon, which is exactly what a recurrent or
            reference-conditioned surrogate needs.

            Both must be wired like any other input, and both need a value
            function before the optimizer is set up -- see
            :py:func:`set_constant_parameters` and :py:func:`set_constant_tvp`.
        z_spec: ``[(name, size), ...]`` for algebraic states (``_z``). Supplying
            this makes the model a **DAE**: do-mpc then requires exactly one
            algebraic equation per algebraic state, given through ``algebraic``.
        algebraic: ``[(key, z_name), ...]`` assigning graph outputs to the
            equations ``0 = g(x, u, z, p, p_tv)``. The expression is used **as
            returned by the network**, so a network that predicts ``z`` directly
            must be wrapped to output ``z_predicted - z`` instead. ``key`` accepts
            an index, a node name, or a ``(key, slice)`` pair.
        measurements: ``[(source, name), ...]`` or
            ``[(source, name, meas_noise), ...]`` declaring measurement equations
            via :py:meth:`do_mpc.model.Model.set_meas`. ``source`` is a graph
            output key (index / name / ``(key, slice)``) or a ``('x', var)`` /
            ``('z', var)`` reference to an existing model variable.
            ``meas_noise`` defaults to ``True``, which creates the corresponding
            ``_v`` variable that :py:class:`do_mpc.estimator.MHE` needs. Without
            any measurement do-mpc assumes full state feedback.
        sample_args: Example inputs for tracing a torch module.
        input_names: Graph input names when exporting a torch module. Should
            match the names used in ``wiring``.
        opset: ONNX opset used when exporting a torch module.
        model_type: ``'discrete'`` (recommended) or ``'continuous'``. Choosing
            ``'continuous'`` emits a warning, see the Note below.
        symvar_type: ``'SX'`` (default) or ``'MX'``. Both work; the placeholders
            used during conversion follow this choice.

    Note:
        **``model_type`` is a semantic claim the builder cannot verify.** With
        ``'discrete'`` the network output is taken as :math:`x_{k+1}`; with
        ``'continuous'`` it is taken as :math:`\dot x` and discretized by
        orthogonal collocation. Nothing in the trained weights distinguishes a
        one-step map from a derivative, so pick this to match how the network was
        trained. A warning is emitted for ``'continuous'``.

    Note:
        A graph output may be **sliced** so that one wide output can drive several
        states, e.g. ``outputs=[(0, 'x'), ((0, slice(2, 4)), 'extra')]``. Any
        queryable node -- not just a declared graph output -- can be used as a
        key; run ``print(converter)`` to list them.
        verbose: Print the resolved wiring.

    Returns:
        tuple: ``(model, converter)`` -- a set-up :py:class:`do_mpc.model.Model`
        and the :py:class:`do_mpc.sysid.ONNXConversion` instance behind it. The
        converter can be queried for any intermediate node, e.g. to build
        additional monitoring expressions with
        :py:meth:`do_mpc.model.Model.set_expression`.

    Raises:
        Exception: if the spec is inconsistent, a graph input cannot be wired, or
            **an output width does not match the state it drives**. The width
            check is what turns an otherwise opaque
            ``Error in powerIndex slicing`` into an actionable message.
    """
    # Checked first: it is cheap, and otherwise a bad `source` would be masked
    # by a spec-validation error (or the other way round).
    if model_type == 'continuous':
        warnings.warn(
            "ann_to_dompc_model is building a CONTINUOUS model. do-mpc will treat the "
            "network output as dx/dt and discretize it with orthogonal collocation. "
            "That is only correct if the network was trained to predict a derivative. "
            "A network trained on (x_k, u_k) -> x_{k+1} pairs must use "
            "model_type='discrete'; the weights themselves carry no indication of which "
            "it is, so this cannot be checked automatically. Collocation also "
            "instantiates the network at n_horizon * collocation_ni * "
            "(collocation_deg + 1) points, which makes the NLP grow quickly.",
            UserWarning, stacklevel=2)

    if not _is_supported_source(source):
        raise Exception('source must be an onnx.ModelProto, a path to a .onnx file, or a '
                        'torch.nn.Module (the latter also needs sample_args). You have: {}'
                        .format(type(source).__name__))

    validate_wiring(x_spec, u_spec, wiring, outputs, p_spec, tvp_spec,
                    z_spec, algebraic, measurements)

    onnx_model = _load_source(source, sample_args, input_names, opset)
    converter = ONNXConversion(onnx_model)

    # ---- 1. check the wiring against the actual graph inputs ----------------
    graph_inputs = list(converter.inputshape)
    wired = [name for name, _ in wiring]
    if sorted(wired) != sorted(graph_inputs):
        raise Exception('wiring covers %s but the ONNX graph has inputs %s. Every graph '
                        'input must be wired exactly once.' % (wired, graph_inputs))

    sizes = {('x', name): int(size) for name, size in x_spec}
    for kind, spec in (('u', u_spec), ('p', p_spec), ('tvp', tvp_spec), ('z', z_spec)):
        sizes.update({(kind, name): int(size) for name, size in spec})
    for onnx_name, parts in wiring:
        expected = sum(sizes[tuple(part)] for part in parts)
        actual = converter.inputshape[onnx_name][-1]
        if expected != actual:
            raise Exception('wiring mismatch for ONNX input %r: it is %d-wide but the '
                            'concatenation of %s is %d-wide.'
                            % (onnx_name, actual, [tuple(p) for p in parts], expected))

    # ---- 2. convert with symbolic placeholders shaped like the graph inputs --
    # The placeholder type must follow symvar_type: mixing MX model variables into
    # an SX expression (or the reverse) fails deep inside CasADi with an opaque
    # "Wrong number or type of arguments for overloaded function".
    symbol_type = casadi.MX if symvar_type == 'MX' else casadi.SX
    symbols = {name: symbol_type.sym(name, 1, shape[-1] if shape[-1] else 1)
               for name, shape in converter.inputshape.items()}
    converter.convert(**{name: symbols[name] for name in graph_inputs})
    output_names = converter.output_layers

    # ---- 3. resolve every graph output and check it against its target ------
    target_sizes = dict(x_spec)
    target_sizes.update(dict(z_spec))

    def resolve(entries, what):
        result = []
        for key, target in entries:
            name, expression = resolve_output(converter, key)
            width = int(expression.shape[1])
            expected = int(target_sizes[target])
            if width != expected:
                raise Exception('wiring mismatch: %s source %r (%s) is %d-wide but do-mpc '
                                '%s %r is %d-wide. Graph outputs are %s.'
                                % (what, name, tuple(expression.shape), width, what,
                                   target, expected,
                                   [(o, tuple(converter[o].shape)) for o in output_names]))
            result.append((name, target, expression))
        return result

    resolved = resolve(outputs, 'state')
    resolved_algebraic = resolve(algebraic, 'algebraic')

    resolved_measurements = []
    for entry in measurements:
        source, meas_name = entry[0], entry[1]
        meas_noise = True if len(entry) < 3 else bool(entry[2])
        if isinstance(source, tuple) and len(source) == 2 \
                and source[0] in ('x', 'u', 'z', 'p', 'tvp') and isinstance(source[1], str):
            # a reference to an existing model variable, not a graph output
            resolved_measurements.append((source, meas_name, None, meas_noise))
        else:
            name, expression = resolve_output(converter, source)
            resolved_measurements.append((name, meas_name, expression, meas_noise))

    if verbose:
        print('ann_to_dompc_model: %d graph inputs, %d outputs, n_p=%d, n_tvp=%d'
              % (len(graph_inputs), len(resolved), len(p_spec), len(tvp_spec)))
        for onnx_name, parts in wiring:
            print('   input  %-14s <- %s' % (onnx_name, [tuple(p) for p in parts]))
        for name, state, expression in resolved:
            print('   output %-14s -> _x[%r] %s' % (name, state, tuple(expression.shape)))
        for name, z_name, expression in resolved_algebraic:
            print('   output %-14s -> 0 = _z[%r] %s' % (name, z_name, tuple(expression.shape)))
        for name, meas_name, expression, noise in resolved_measurements:
            print('   meas   %-14s -> _y[%r] noise=%s' % (name, meas_name, noise))

    # ---- 4. declare the do-mpc variables ------------------------------------
    model = do_mpc.model.Model(model_type, symvar_type)
    for name, size in x_spec:
        model.set_variable('_x', name, shape=(int(size), 1))
    for name, size in u_spec:
        model.set_variable('_u', name, shape=(int(size), 1))
    for name, size in p_spec:
        model.set_variable('_p', name, shape=(int(size), 1))
    for name, size in tvp_spec:
        model.set_variable('_tvp', name, shape=(int(size), 1))
    for name, size in z_spec:
        model.set_variable('_z', name, shape=(int(size), 1))

    columns = {'x': {name: model.x[name] for name, _ in x_spec},
               'u': {name: model.u[name] for name, _ in u_spec},
               'p': {name: model.p[name] for name, _ in p_spec},
               'tvp': {name: model.tvp[name] for name, _ in tvp_spec},
               'z': {name: model.z[name] for name, _ in z_spec}}

    # ---- 5. substitute the do-mpc variables into the converted expressions --
    # ONNX tensors are row vectors (1, n), do-mpc variables are column vectors
    # (n, 1), so both the placeholders and the results are transposed here.
    def to_column(expression):
        column_expression = expression.T
        for onnx_name, parts in wiring:
            target = casadi.vertcat(*[columns[kind][var] for kind, var in parts]).T
            column_expression = casadi.substitute(column_expression,
                                                  symbols[onnx_name], target)
        return column_expression

    for _, state, expression in resolved:
        model.set_rhs(state, to_column(expression))

    for _, z_name, expression in resolved_algebraic:
        model.set_alg(z_name, to_column(expression))

    for source, meas_name, expression, meas_noise in resolved_measurements:
        if expression is None:
            kind, var = source
            model.set_meas(meas_name, columns[kind][var], meas_noise=meas_noise)
        else:
            model.set_meas(meas_name, to_column(expression), meas_noise=meas_noise)

    model.setup()
    return model, converter


def resolve_output(converter: ONNXConversion, key, wiring_widths: Optional[Dict[str, int]] = None):
    """Resolve one ``outputs`` / ``algebraic`` / ``measurements`` entry.

    Accepted forms:

    * ``int`` -- index into ``converter.output_layers``;
    * ``str`` -- an ONNX graph output **or any intermediate node name** (see
      ``print(converter)`` for the queryable keys);
    * ``(key, slice)`` -- a sub-range of the selected tensor, so a single wide
      graph output can drive several states, or only part of it can be used.

    Args:
        converter: A converter that has already run :py:meth:`convert`.
        key: One of the forms above.
        wiring_widths: Unused; accepted for signature symmetry.

    Returns:
        tuple: ``(name, expression)`` where ``expression`` is a CasADi **row**
        vector taken from the graph.

    Raises:
        Exception: if the key cannot be resolved or the slice is out of range.
    """
    if isinstance(key, tuple):
        if len(key) != 2:
            raise Exception('an outputs entry may be key or (key, slice); got %r' % (key,))
        key, selector = key
    else:
        selector = None

    names = converter.output_layers
    if isinstance(key, int):
        if not 0 <= key < len(names):
            raise Exception('output index %d is out of range; the graph has %d outputs: %s'
                            % (key, len(names), names))
        name = names[key]
    elif isinstance(key, str):
        if key in names:
            name = key
        else:
            try:
                converter[key]
            except Exception:
                raise Exception('%r is neither a graph output (%s) nor a queryable '
                                'intermediate node. Print the converter to list them.'
                                % (key, names))
            name = key
    else:
        raise Exception('outputs keys must be an int, a str, or a (key, slice) pair; '
                        'got %r' % (key,))

    expression = converter[name]
    if selector is not None:
        width = int(expression.shape[1])
        # Validate before indexing: CasADi raises its own sparsity assertion
        # ("Slice (start=5, stop=9, step=1) out of bounds") for an out-of-range
        # slice, which does not name the offending spec entry.
        if not isinstance(selector, slice):
            raise Exception('the second element of %r must be a slice' % (key,))
        start, stop, step = selector.indices(width)
        if start >= stop or stop > width:
            raise Exception('slice %r selects nothing from the %d-wide output %r '
                            '(resolved to [%d:%d])' % (selector, width, name, start, stop))
        expression = expression[:, start:stop:step]
    return name, expression


def _template_index_form(template, kind: str, name: str) -> Tuple:
    """Return the power index that addresses **every** entry of ``template[kind][name]``.

    The template shape differs between do-mpc classes, which is the single most
    confusing part of supplying parameter values:

    * :py:class:`do_mpc.controller.MPC` templates carry an **extra leading
      dimension** -- the scenario index for ``_p`` and the time index for
      ``_tvp``. Only ``template['_p', :, name]`` works; ``template[name]`` raises
      ``Error occured in struct context with powerIndex (name,)``.
    * :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.estimator.MHE` and
      :py:class:`do_mpc.estimator.EKF` templates are flat, so ``template[name]``
      works and ``template['_p', :, name]`` does not.

    The form is detected from the structure labels rather than from the class, so
    any optimizer that follows either convention is handled.

    Args:
        template: A structure returned by ``get_p_template`` / ``get_tvp_template``.
        kind: ``'_p'`` or ``'_tvp'``.
        name: Variable name as declared in the model.

    Returns:
        tuple: A power index suitable for ``template[index] = value``.
    """
    try:
        labels = template.labels()
    except Exception:
        labels = []
    if labels and str(labels[0]).startswith('[%s,' % kind):
        return (kind, slice(None), name)
    return (name, slice(None))


def _call_template_getter(getter, n_combinations):
    """Call ``get_p_template`` with or without ``n_combinations`` as appropriate.

    Only :py:meth:`do_mpc.controller.MPC.get_p_template` takes the number of
    uncertainty combinations; the Simulator, MHE and EKF versions take none.
    """
    parameters = inspect.signature(getter).parameters
    if parameters:
        return getter(n_combinations)
    return getter()


def _assign_constant(template, kind: str, name: str, value, repeat: int) -> None:
    """Write ``value`` into every slot that ``template[kind][name]`` spans.

    The CasADi structure setter does **not** broadcast: assigning a length-2 list
    to a range that spans 3 scenarios x 2 parameters raises
    ``Rhs out of range. Got list index 2 but rhs list is only of length 2``.
    So for the MPC-style templates, which carry a leading scenario / horizon
    dimension, every index along that dimension is filled explicitly.

    Args:
        template: A ``get_p_template`` / ``get_tvp_template`` result.
        kind: ``'_p'`` or ``'_tvp'``.
        name: Variable name.
        value: Scalar or array-like matching the variable's size.
        repeat: Length of the leading dimension (1 for flat templates).
    """
    index = _template_index_form(template, kind, name)
    try:
        if index[0] == kind:
            for position in range(max(1, int(repeat))):
                template[(kind, position, name)] = value
        else:
            template[index] = value
    except Exception as exc:
        # The common cause is an MHE: parameters listed in ``p_est_list`` are
        # being estimated, so they are deliberately absent from the p template --
        # only the NON-estimated parameters go through set_p_fun. Without this
        # the user sees a bare casadi.tools.structure "Unknown keyword" error.
        try:
            available = sorted(template.keys())
        except Exception:
            available = []
        raise Exception(
            "cannot assign %s %r: it is not present in this optimizer's template. "
            "Available: %s. For an MHE, parameters listed in p_est_list are being "
            "estimated and must NOT be supplied here -- pass only the non-estimated "
            "ones. (casadi said: %s)"
            % (kind, name, available or '(empty)', str(exc).splitlines()[0][:120]))


def set_constant_parameters(optimizer: Any, n_combinations: int = 1, **values):
    """Register a ``p_fun`` that returns the same parameter values at every call.

    Note:
        For an :py:class:`do_mpc.estimator.MHE`, only the parameters that are
        **not** being estimated belong in the template -- the ones named in
        ``p_est_list`` are optimization variables. Supplying an estimated
        parameter here raises an explanatory error instead of a bare
        ``casadi.tools.structure`` "Unknown keyword" message. If every parameter
        is estimated the template is empty and this function should not be called
        at all; set ``p_fun`` only for the remaining ones.

    This removes the boilerplate around :py:meth:`do_mpc.model.Model` parameters
    (``_p``), whose template shape differs between the controller and the other
    classes -- see :py:func:`_template_index_form`.

    Args:
        optimizer: An object exposing ``get_p_template`` and ``set_p_fun``, i.e.
            an :py:class:`do_mpc.controller.MPC`,
            :py:class:`do_mpc.simulator.Simulator`,
            :py:class:`do_mpc.estimator.MHE` or :py:class:`do_mpc.estimator.EKF`.
        n_combinations: Number of uncertainty scenarios. Only used by
            :py:class:`do_mpc.controller.MPC`; ignored elsewhere. Passing more
            than one and filling every scenario with the same value gives the
            nominal (non-robust) case -- for genuine robust MPC supply the
            per-scenario values through your own ``p_fun`` or use
            :py:meth:`do_mpc.controller.MPC.set_uncertainty_values`.
        **values: ``name=value`` pairs for the parameters declared in the model.

    Returns:
        The filled template, so the caller can inspect or further modify it.

    Example:
        ::

            model, conv = do_mpc.sysid.ann_to_dompc_model(
                net, x_spec=[('x', 2)], u_spec=[('u', 1)], p_spec=[('theta', 2)],
                wiring=[('xu', [('x', 'x'), ('u', 'u'), ('p', 'theta')])],
                outputs=[(0, 'x')], ...)

            mpc = do_mpc.controller.MPC(model); ...
            do_mpc.sysid.set_constant_parameters(mpc, theta=[1.0, 0.5])
            mpc.setup()
    """
    template = _call_template_getter(optimizer.get_p_template, n_combinations)
    for name, value in values.items():
        # For an MPC the leading dimension is the uncertainty scenario, so the
        # same value has to be written once per scenario.
        _assign_constant(template, '_p', name, value, n_combinations)
    optimizer.set_p_fun(lambda t_now: template)
    return template


def set_constant_tvp(optimizer: Any, **values):
    """Register a ``tvp_fun`` that returns the same values over the whole horizon.

    For :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.MHE` the
    template spans the entire prediction / estimation horizon, so a single value
    is broadcast to every step -- the usual way to express a **constant setpoint**.
    For a genuinely time-varying reference, write your own ``tvp_fun`` and index
    the horizon explicitly (``template[name, k]`` for the MPC form).

    Args:
        optimizer: An object exposing ``get_tvp_template`` and ``set_tvp_fun``.
        **values: ``name=value`` pairs for the ``_tvp`` declared in the model.

    Returns:
        The filled template.
    """
    template = optimizer.get_tvp_template()
    horizon = getattr(getattr(optimizer, 'settings', None), 'n_horizon', None)
    repeat = 1 if horizon is None else int(horizon) + 1
    for name, value in values.items():
        # For an MPC / MHE the leading dimension is the prediction or estimation
        # horizon, so the same value has to be written once per step.
        _assign_constant(template, '_tvp', name, value, repeat)
    optimizer.set_tvp_fun(lambda t_now: template)
    return template
