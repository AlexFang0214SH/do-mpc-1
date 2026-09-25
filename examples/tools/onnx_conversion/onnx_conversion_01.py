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
"""ONNX -> CasADi conversion, example 1: a single-input Keras network.

Demonstrates the smallest complete path from a trained framework model to a
CasADi expression that can be embedded in ``do_mpc.model.Model``:

    1. build / train a Keras model;
    2. export it to ONNX with ``tf2onnx``;
    3. hand the ``onnx.ModelProto`` to ``do_mpc.sysid.ONNXConversion``;
    4. call ``convert`` with either numpy values or CasADi symbols;
    5. query any layer by name to get its CasADi expression.

Note that ``ONNXConversion`` takes an **ONNX model**, never a framework model.
The framework -> ONNX step is the exporter's job (``tf2onnx`` here,
``torch.onnx.export`` for PyTorch).

Extra dependencies (NOT part of ``pip install do-mpc[full]``)::

    pip install tensorflow tf2onnx

Run from this directory::

    python onnx_conversion_01.py

See ``onnx_conversion_02.py`` for a multi-input graph with Concat / Add / Slice.
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

OPSET = 17


def keras_to_onnx(keras_model, sample_inputs):
    """Export a Keras model to ONNX and return the ``onnx.ModelProto``.

    ``sample_inputs`` is a single array for a one-input model and a list of
    arrays for a multi-input model -- exactly what ``keras_model(...)`` expects.
    """
    # Keras 3 refuses to export a model that has never been called.
    keras_model(sample_inputs)
    signature = [tf.TensorSpec(tensor.shape, tf.float32, name=tensor.name.split(':')[0])
                 for tensor in keras_model.inputs]
    # No output_path: from_keras returns the ModelProto directly, so the
    # example never touches the filesystem.
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=signature, opset=OPSET)
    return model_proto


def main():
    # ------------------------------------------------------------ 1. the model
    model_input = keras.Input(shape=(3,), name='input')
    hidden_layer = keras.layers.Dense(5, activation='relu', name='hidden')(model_input)
    output_layer = keras.layers.Dense(1, activation='linear', name='output')(hidden_layer)
    keras_model = keras.Model(inputs=model_input, outputs=output_layer)

    # ------------------------------------------------------- 2. export to ONNX
    onnx_model = keras_to_onnx(keras_model, np.zeros((1, 3), dtype='float32'))
    ops = sorted({node.op_type for node in onnx_model.graph.node})
    print('ONNX graph: %d nodes, ops = %s' % (len(onnx_model.graph.node), ops))

    # --------------------------------------------------------- 3. the converter
    converter = do_mpc.sysid.ONNXConversion(onnx_model)

    # print(converter) lists the input names/shapes and every queryable layer name
    print()
    print(converter)

    # ------------------------------ 4a. convert with a NUMERIC (numpy) input --
    numeric_input = np.ones((1, 3), dtype='float32')
    converter.convert(input=numeric_input)
    print('numeric  convert -> converter["output"] =')
    print('   ', np.asarray(converter['output']).ravel())

    # --------------------------- 4b. convert with a SYMBOLIC (CasADi) input ---
    # Re-running convert() replaces the previously stored node values, so the same
    # instance can be reused with a different input type.
    import casadi
    x_symbolic = casadi.SX.sym('x', 1, 3)
    converter.convert(input=x_symbolic)
    expression = converter['output']
    print()
    print('symbolic convert -> converter["output"] =')
    print('   ', expression)

    # ------------------------------------------------- 5. validate against Keras
    # Building the CasADi Function from the SAME symbolic object that was passed
    # to convert() matters: CasADi >= 3.8 matches symbols by identity, not name.
    casadi_function = casadi.Function('f', [x_symbolic], [expression])
    test_input = np.array([[1.0, 2.0, 3.0]], dtype='float32')
    casadi_value = np.array(casadi.DM(casadi_function(test_input)).full())
    keras_value = np.asarray(keras_model(test_input, training=False))

    print()
    print('input          :', test_input.ravel())
    print('Keras output   :', keras_value.ravel())
    print('CasADi output  :', casadi_value.ravel())
    print('max |Keras - CasADi| = %.2e' % np.abs(keras_value - casadi_value).max())

    # --------------------------------------- 6. what you would do with it next
    print()
    print('The expression above is an ordinary CasADi value, so it can be used')
    print('directly in a do-mpc model, e.g.:')
    print('    model.set_rhs("my_state", converter["output"])')


if __name__ == '__main__':
    main()