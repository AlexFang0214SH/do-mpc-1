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
r"""ONNX -> CasADi conversion, example 2: a multi-input Keras graph.

``onnx_conversion_01.py`` covers the minimal single-input case. This example uses
a deliberately richer graph so that the operators a real network produces are all
exercised:

    first_input (1,3) --+
                        +--> Concat(axis=1) --> Dense(tanh) ----+--> t1
    second_input (1,2)--+                    \-> Dense(relu) ---+--> r2
                                                                 |
                                              Add(t1, r2) -------+
                                                     |
                                        Lambda(x[:, :2])  ->  ONNX Slice
                                                     |
                                          Dense(tanh) -> model_output

Operators involved: ``Concat``, ``MatMul``/``Gemm``, ``Tanh``, ``Relu``, ``Add``,
``Slice`` -- all supported by ``do_mpc.sysid.ONNXOperations``.

It also demonstrates the three input modes ``convert`` accepts:

    * CasADi symbols  -> yields a symbolic expression you can differentiate and
                         embed in a ``do_mpc.model.Model``;
    * numpy values    -> yields plain numbers, handy for a quick sanity check;
    * a mix of both   -> partially evaluates the graph, which is useful when some
                         inputs are known constants (e.g. a measured disturbance)
                         and only the rest should stay symbolic.

Note on ``Slice``: a Keras ``Lambda(lambda x: x[:, :2])`` exports to an ONNX
``Slice`` node whose ``starts``/``ends``/``axes`` are **inputs**, not attributes
(the opset >= 10 convention). Every current exporter produces that form.

Extra dependencies (NOT part of ``pip install do-mpc[full]``)::

    pip install tensorflow tf2onnx

Run from this directory::

    python onnx_conversion_02.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join('..', '..', '..'))
import do_mpc

# ---------------------------------------------------------------- dependencies
try:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')   # silence TF C++ logs
    import tensorflow as tf
    from tensorflow import keras
    import tf2onnx
    import onnx
except ImportError as exc:
    sys.exit(
        'This example needs tensorflow and tf2onnx, which are not part of\n'
        'do-mpc\'s dependencies. Install them with:\n\n'
        '    pip install tensorflow tf2onnx\n\n'
        '(missing module: {})'.format(exc.name))

import casadi

OPSET = 17


def keras_to_onnx(keras_model, sample_inputs):
    """Export a Keras model to ONNX and return the ``onnx.ModelProto``."""
    keras_model(sample_inputs)          # Keras 3 requires the model to be called once
    signature = [tf.TensorSpec(tensor.shape, tf.float32, name=tensor.name.split(':')[0])
                 for tensor in keras_model.inputs]
    # No output_path: from_keras returns the ModelProto directly, so the
    # example never touches the filesystem.
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=signature, opset=OPSET)
    return model_proto


def build_keras_model():
    """The multi-input graph described in the module docstring."""
    model_input1 = keras.Input(shape=(3,), name='first_input')
    model_input2 = keras.Input(shape=(2,), name='second_input')

    concat_layer = keras.layers.concatenate([model_input1, model_input2],
                                            name='concatenation_layer', axis=1)
    hidden_layer = keras.layers.Dense(5, name='hidden_layer', activation='tanh')(concat_layer)
    hidden_layer2 = keras.layers.Dense(5, name='hidden_layer2', activation='relu')(hidden_layer)
    sum_layer = keras.layers.add([hidden_layer, hidden_layer2], name='sum_layer')
    slice_layer = keras.layers.Lambda(lambda x: x[:, :2], name='slice_layer')(sum_layer)
    model_output = keras.layers.Dense(5, name='model_output', activation='tanh')(slice_layer)

    return keras.Model(inputs=[model_input1, model_input2],
                       outputs=model_output, name='model')


def main():
    model = build_keras_model()

    # ------------------------------------------------------- export to ONNX
    test_inp1 = np.array([1.0, 2.0, 3.0], dtype='float32').reshape(1, -1)
    test_inp2 = np.array([2.0, 2.0], dtype='float32').reshape(1, -1)
    onnx_model = keras_to_onnx(model, [np.zeros((1, 3), 'float32'), np.zeros((1, 2), 'float32')])
    print('ONNX graph: %d nodes, ops = %s'
          % (len(onnx_model.graph.node), sorted({n.op_type for n in onnx_model.graph.node})))
    print('ONNX graph outputs: %s' % [o.name for o in onnx_model.graph.output])

    converter = do_mpc.sysid.ONNXConversion(onnx_model)

    # print(converter) lists the input names/shapes and every queryable key.
    # Only the GRAPH OUTPUT keeps its Keras layer name ('model_output'); the
    # intermediate nodes are named by the exporter, e.g.
    # 'model_1/sum_layer_1/Add:0'. Use print(converter) to find them.
    print()
    print(converter)

    keras_output = np.asarray(model([test_inp1, test_inp2], training=False))

    # ============================== mode 1: CasADi symbolic inputs ============
    input1 = casadi.SX.sym('in1', 1, 3)
    input2 = casadi.SX.sym('in2', 1, 2)
    converter.convert(first_input=input1, second_input=input2, verbose=False)
    casadi_expression = converter['model_output']

    # Build the Function from the SAME symbolic objects passed to convert():
    # CasADi >= 3.8 matches symbols by identity, not by name.
    casadi_function = casadi.Function('casadi_function', [input1, input2], [casadi_expression])
    symbolic_value = np.array(casadi.DM(casadi_function(test_inp1, test_inp2)).full())

    print('--- mode 1: both inputs symbolic ---')
    print('  expression shape :', casadi_expression.shape)
    print('  Keras  output    :', keras_output.ravel())
    print('  CasADi output    :', symbolic_value.ravel())
    print('  max |Keras - CasADi| = %.2e' % np.abs(keras_output - symbolic_value).max())

    # ============================== mode 2: plain numpy inputs ================
    converter.convert(first_input=test_inp1, second_input=test_inp2, verbose=False)
    numeric_output = np.asarray(converter['model_output'])

    print('--- mode 2: both inputs numeric ---')
    print('  CasADi output    :', numeric_output.ravel())
    print('  max |Keras - CasADi| = %.2e' % np.abs(keras_output - numeric_output).max())

    # ============================== mode 3: mixed inputs ======================
    # 'second_input' is treated as a known constant; only 'first_input' stays
    # symbolic. This is the pattern to use when part of the network input is a
    # measured disturbance and the rest is an optimization variable.
    converter.convert(first_input=input1, second_input=test_inp2, verbose=False)
    mixed_expression = converter['model_output']
    mixed_function = casadi.Function('mixed_function', [input1], [mixed_expression])
    mixed_output = np.array(casadi.DM(mixed_function(test_inp1)).full())

    print('--- mode 3: first_input symbolic, second_input numeric ---')
    print('  CasADi output    :', mixed_output.ravel())
    print('  max |Keras - CasADi| = %.2e' % np.abs(keras_output - mixed_output).max())

    # The partially evaluated expression is still differentiable w.r.t. the
    # remaining symbolic input -- this is what makes it usable inside an MPC.
    jacobian = casadi.Function('j', [input1], [casadi.jacobian(mixed_expression, input1)])
    print('  d(output)/d(first_input) shape:', np.array(casadi.DM(jacobian(test_inp1)).full()).shape)

    # ============================== summary ===================================
    worst = max(float(np.abs(keras_output - symbolic_value).max()),
                float(np.abs(keras_output - numeric_output).max()),
                float(np.abs(keras_output - mixed_output).max()))
    print()
    print('worst deviation over all three modes = %.2e' % worst)
    print('Intermediate nodes are queryable too, but under exporter-generated')
    print('names (e.g. "model_1/sum_layer_1/Add:0") -- see print(converter) above.')



if __name__ == '__main__':
    main()