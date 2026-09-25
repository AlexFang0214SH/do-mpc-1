import casadi
import onnx
from onnx import numpy_helper, AttributeProto
import numpy as np
import pdb
import importlib
from typing import List, Dict, Tuple, Union, Callable, Any, Optional


class _SplitRequest:
    """Marker returned by :py:meth:`ONNXOperations.Split`.

    ``Split`` is the only supported operator whose number of outputs cannot be
    derived from its inputs alone: ONNX allows the ``split`` input to be omitted,
    in which case the tensor is divided into as many *equal* parts as the node
    declares outputs. Only :py:meth:`ONNXConversion.convert` knows that number,
    so ``Split`` returns this marker and ``convert`` performs the slicing.

    Args:
        axis: Axis along which to split.
        sizes: Size of every part, or ``None`` to request an equal-sized split.
        total: Total extent of ``axis`` in the input tensor.
    """

    __slots__ = ('axis', 'sizes', 'total')

    def __init__(self, axis: int, sizes: Optional[List[int]], total: int):
        self.axis = axis
        self.sizes = sizes
        self.total = total


class ONNXConversion:
    """ Transform `ONNX model <https://onnx.ai>`_. 
    The transformation returns a CasADi expression of the model and can be used e.g. in the :py:class:`do_mpc.model.Model` class.

    Warning:
        The feature is experimental and currently only has a limited number of supported operations.
        All supported operations can be found in the :py:class:`ONNXOperations` class.

        Other known limitations are listed at the end of this page.

    Note:
        **Recurrent networks.** CasADi matrices are strictly two-dimensional, whereas the
        fused ONNX ``LSTM``, ``GRU`` and ``RNN`` operators work on three-dimensional
        tensors of shape ``[seq_length, num_directions*batch_size, hidden_size]``.
        These operators are therefore **not** supported and cannot be.

        Export the recurrent **cell** instead of the sequence module. A cell has no ONNX
        counterpart, so the exporting framework decomposes it into primitive operators
        that this class does support. With PyTorch, exporting ``torch.nn.LSTMCell``
        yields only ``Add``, ``Constant``, ``Gemm``, ``Mul``, ``Sigmoid``, ``Split`` and
        ``Tanh``, while exporting ``torch.nn.LSTM`` yields a fused three-dimensional
        ``LSTM`` operator plus a shape-juggling sub-graph::

            # do this
            torch.onnx.export(torch.nn.LSTMCell(n_in, n_hid), args, 'cell.onnx',
                              dynamo = False, opset_version = 17)

            # not this
            torch.onnx.export(torch.nn.LSTM(n_in, n_hid), args, 'seq.onnx', ...)

        The cell formulation is also what a model predictive controller needs: MPC rolls
        the model forward over the prediction horizon, so the hidden state should be
        declared as a state of :py:class:`do_mpc.model.Model` and the single-step
        recurrence supplied through :py:meth:`do_mpc.model.Model.set_rhs`. See the
        ``examples/lstm_surrogate_model`` example.

        Note that ``torch.onnx.export`` must be called with ``dynamo = False``: the
        dynamo-based exporter emits operators that are not implemented here.
    
    **How to use:** 

    1. Create an ONNX model in your favorite framework (e.g. `TensorFlow <https://www.tensorflow.org/>`_, `PyTorch <https://pytorch.org/>`_, `Keras <https://keras.io/>`_, `ONNX <https://onnx.ai/>`_).
    
    2. Initiate the :py:class:`ONNXConversion` class with the ONNX model as input.

    3. Obtain information about model inputs and ouputs by printing the class instance.

    4. Call the :py:meth:`ONNXConversion.convert` method, passing with keyword arguments the external inputs of the model. The inputs are propagated through the model and all node expressions are created.

    5. Query the class instance with the respective layer or node name to obtain the CasADi expression of the respective layer or node.


    **Example:**

    We start with a simple Tensorflow (with Keras) model:
    ::
    
        model_input = keras.Input(shape=(3), name='input')
        hidden_layer = keras.layers.Dense(5, activation='relu', name='hidden')(model_input)
        output_layer = keras.layers.Dense(1, activation='linear', name='output')(hidden_layer)

        keras_model = keras.Model(inputs=model_input, outputs=output_layer)

    We then proceed to export the model in the ONNX format, using the `tf2onnx <https://pypi.org/project/tf2onnx/>`_ package:

    ::

        model_input_signature = [
            tf.TensorSpec(np.array((1, 3)), name='input'),
        ]
        output_path = os.path.join('models', 'model.onnx')

        onnx_model, _ = tf2onnx.convert.from_keras(keras_model, 
            output_path=output_path, 
            input_signature=model_input_signature
        )

    We can now use the ONNX model (either directly or loaded from disc) to initialize the :py:class:`ONNXConversion` class:

    ::

        casadi_converter = do_mpc.sysid.ONNXConversion(onnx_model)

    Obtain information about the model inputs and outputs by calling ``print(casadi_converter)``, yielding, in this example:

    .. code-block:: console

        ONNX2Casadi model 'casadi_model' 
        ----------------------------------
        Call 'convert' by supplying the inputs with respective name and shape below.
        Input shape of 'input' is (1, 3)
        ----------------------------------
        Query the instance with the following keywords to obtain the CasADi expression of the respective layer or graph operation node:
        - 'input'
        - 'model_4/hidden/MatMul:0'
        - 'model_4/hidden/Relu:0'
        - 'output'

    Call the :py:meth:`ONNXConversion.convert` method, considering the name and shape of the inputs:

    :: 

        # Inputs can be numpy arrays
        casadi_converter.convert(input=np.ones((1,3)))

        # or CasADi expressions
        x = casadi.SX.sym('x',1,3)
        casadi_converter.convert(input=x)

    Query the instance with the respective layer or node name to obtain the CasADi expression of the respective layer or node:

    ::

        print(casadi_converter['output'])

    Args:
        model: An ONNX model.
        model_name: Name of the model

    """
    
    def __init__(self, model: onnx.onnx_ml_pb2.ModelProto, model_name: Optional[str]=None):  
        # In case of a keras model as input, convert it to an ONNX model
        
        if isinstance(model,(onnx.onnx_ml_pb2.ModelProto)):
            self.onnx_model = model
            self.name = "casadi_model" if not isinstance(model_name, (str)) else model_name
        else:
            raise Exception("Please pass an ONNX model (onnx.ModelProto) as input, e.g. "
                            "the result of onnx.load('model.onnx'). Framework models must be "
                            "exported to ONNX first: use torch.onnx.export(...) for PyTorch, or "
                            "tf2onnx.convert.from_keras(...) / model.export(format='onnx') for "
                            "Keras and TensorFlow.")
        
        # From the ONNX model the graph and the nodes and the initializers are directly inherited
        self.graph = self.onnx_model.graph
        self.nodes = list(self.graph.node)
        onnx_initializers = list(self.graph.initializer)
        
        # The initialized tensors are converted into the numpy readable format  before assignment
        self.initialized_tensors = {}
        for initializer in onnx_initializers:
            self.initialized_tensors[initializer.name] = numpy_helper.to_array(initializer)
        
            
        # Determining the input shape 
        self.inputshape = {}
        for inpn in self.graph.input:
            if inpn.name not in self.initialized_tensors.keys():
                self.inputshape[inpn.name] = tuple([shape_dim.dim_value for shape_dim in inpn.type.tensor_type.shape.dim])
        
         
        # Determining output layer names
        self.output_layers = [out.name for out in self.graph.output]
        self.layers = [n.name for n in list(self.graph.input)] + [n.output[0] for n in self.nodes]
        

        # Rank table: ONNX tensors may have any rank while CasADi is strictly
        # two-dimensional, so _to_casadi_shape drops leading singleton dimensions.
        # Axis-referencing operators need to know how many were dropped, otherwise
        # a positive axis silently addresses the wrong dimension. Shape inference
        # is best-effort: when it fails the offset defaults to 0, which reproduces
        # the plain rank-2 behaviour.
        self.ranks = self._collect_ranks(self.onnx_model)

        # Create instance of operations class
        self.operations = ONNXOperations()
        

    @staticmethod
    def _collect_ranks(model) -> Dict[str, int]:
        """Map every known tensor name in ``model`` to its ONNX rank.

        Uses :func:`onnx.shape_inference.infer_shapes` so intermediate values are
        covered too, not just the graph inputs and outputs. Any failure (an
        unsupported operator, a model too large for the in-memory API) degrades to
        whatever ranks are already known; names that are still missing are treated
        as rank 2 by :py:meth:`_axis_offset`, which reproduces the previous
        behaviour.
        """
        ranks = {}

        def note(name, dims):
            if name and dims:
                ranks[name] = len(dims)

        def scan(graph):
            for value_info in (list(graph.input) + list(graph.value_info)
                               + list(graph.output)):
                note(value_info.name, list(value_info.type.tensor_type.shape.dim))
            for initializer in graph.initializer:
                note(initializer.name, list(initializer.dims))

        try:
            scan(model.graph)
        except Exception:
            pass
        try:
            scan(onnx.shape_inference.infer_shapes(model).graph)
        except Exception:
            pass
        return ranks

    def _axis_offset(self, name: str) -> int:
        """Number of leading dimensions dropped when ``name`` was stored in CasADi."""
        if not name:
            return 0
        rank = self.ranks.get(name)
        return 0 if rank is None else max(0, int(rank) - 2)

    def __repr__(self) -> str:
        """ Prints information about the converter.

        Use this method to obtain information about the model inputs and outputs. 
        """

        # Create message
        repr_message = "ONNX2Casadi model '{}' \n".format(self.name)
        repr_message += "----------------------------------\n"
        repr_message += "Call 'convert' by supplying the inputs with respective name and shape below.\n"
        for name, shape in self.inputshape.items():
            repr_message += "Input shape of '{}' is {}\n".format(name, shape)
        repr_message += "----------------------------------\n"
        repr_message += "Query the instance with the following keywords to obtain the CasADi expression of the respective layer or graph operation node:\n"
        for name in self.layers:
            repr_message += " - '{}'\n".format(name)

        return repr_message
        
        

    def _determine_shape(self,raw_shape):
        #TODO: I am not sure we need this. In any case this should be a private method. The user wont activate it.
        """ This method helps to determine the relevant array shape from a given
        ambiguous shape representation.
        
        *Example:*
        
        ::
            
            
        (None,1) and (1, None) as input return (1,)
        (n,m) shapes with n and m not "None" values stays the same
        [n,m] returns (n,m) in a tuple representation
        """
        shape = []
        for dimension in raw_shape:
            #if dimension != None:
            shape.append(dimension)
        return tuple(shape)
    
    
            
    
    def convert(self, verbose=False, **kwargs) -> None:
        """ Evaluate ONNX model with inputs of type ``casadi.SX``, ``casadi.MX``, ``casadi.DM`` or ``numpy.ndarray``.

        The keyword arguments of this method refer to the names of the inputs of the model. 
        If these names are unknown, print the instance of the class to obtain the names.

        Convert does not return anything. The converted model is stored in the instance of the class.
        To obtain the results of the conversion at an arbitrary internal layer, query the instance with the respective layer name.
        Layer names can be obtained by printing the instance of the class.

        Args:
            verbose: If True, prints the conversion progress.
            **kwargs: Keyword arguments of the method refer to the names of the inputs of the model. The values of the keyword arguments are the inputs of the model and can be of type ``casadi.SX``, ``casadi.MX``, ``casadi.DM`` or ``numpy.ndarray``.
        """
        
        
        # Rename for shorther notation
        graph = self.graph
        nodes = self.nodes
        init_tensors = self.initialized_tensors
        inputshape = self.inputshape
        
            
        # Sanity check: Right number of inputs?
        if len(kwargs) != len(inputshape):
            raise Exception("The model takes {} inputs for the layers {}".format(len(inputshape.keys()),list(inputshape.keys())))

        # Sanity check: Right names for inputs
        if not all( layer_name in list(kwargs.keys()) for layer_name in list(inputshape.keys()) ):
            raise Exception("False input layer names.\n The input layers are {}".format(list(inputshape.keys())))

        # Sanity check: Right type for inputs
        if not all(isinstance(value,(casadi.SX,casadi.MX, casadi.DM, np.ndarray)) for value in kwargs.values()):
            raise Exception("Wrong input type. Please pass a CasADi variable or numpy array as input.")

        # Create dict "input" and check the shape of the inputs
        self.input = {}   

        for input_name, shape in inputshape.items():
            # TODO: Write comments on all checks
            check_1 = (len(inputshape[input_name]) == 1)
            check_2 = (inputshape[input_name][0] != kwargs[input_name].shape[0])
            check_3 = (inputshape[input_name] != kwargs[input_name].shape)
            if check_1 and check_2 and check_3:
                raise Exception("The shape of the input '{}' should be {}".format(input_name,inputshape[input_name]))

            self.input[input_name] = kwargs[input_name]



        # Computation of all node values
        node_values = self.input.copy() # "node_values" contains only input node values as initial values
        all_values = self.input.copy()
        all_values.update(init_tensors) # "all_values" contains initializer and input node values as initial values
        
        # Iterate over all nodes
        for n in nodes:
            if verbose:
                print("\nProcessing of {}".format(n.name))
                
            ins = [] # "ins" collects all the input variables for the corresponding
                     # node in CasADi-form with corrected shape (in case input shape is (1,))
                     # These inputs could either be the input values from the last layer
                     # or the bias and weight values, which are saved in "init_tensors"
                     # "and all_values"
                     # Computation follows the node order given by the ONNX graph.
                     # Computed values from the previous node are initialized in "node_values"
                     # These are as well saved in "all_values" in addition to bias and weight values
                     # A computational node is different from a neural layer:
                     # ONNX graph reserves for each mathematical operation a separate
                     # node (bias addition and weight multiplication are 2 nodes) 
                     
            for input_layer_name in n.input:

                # Optional ONNX inputs are encoded as an empty string, e.g. the
                # "sequence_lens" slot of a fused LSTM node. Passing None keeps the
                # positional argument list aligned with the operator signature.
                if input_layer_name == '':
                    ins.append(None)
                    continue

                if input_layer_name not in all_values:
                    raise Exception("Node '{}' ({}) references the unknown input '{}'. "
                                    "The ONNX graph is expected to be topologically sorted."
                                    .format(n.name, n.op_type, input_layer_name))

                # Conversion into CasADi and shape correction in case of (1,) as input shape
                if isinstance(all_values[input_layer_name],(np.ndarray)):
                    pass
                    #all_values[input_layer_name] = casadi.DM(np.atleast_2d(all_values[input_layer_name]))

                ins.append(all_values[input_layer_name]) # critical ! input_layer_name should be already contained in all_values => Assumption: ONNX graph representation is correctly arranged


            # Tell the operator how many leading dimensions were dropped, so it
            # can map an ONNX axis onto a CasADi axis. The input offset applies to
            # rank-preserving operators and to Squeeze (whose axes refer to the
            # input); the output offset applies to Unsqueeze (whose axes refer to
            # the result). Unknown names default to rank 2, i.e. offset 0.
            self.operations._axis_offset_in = self._axis_offset(
                n.input[0] if len(n.input) else '')
            self.operations._axis_offset_out = self._axis_offset(
                n.output[0] if len(n.output) else '')

            # Determination of the operation type and subsequent computation
            if hasattr(self.operations, n.op_type):
                out = getattr(self.operations, n.op_type)(*ins, attribute=n.attribute)
            else:
                raise Exception("Operation '{}' not implemented. Please consider the limited set of operations available to the tool.".format(n.op_type))

            # A node may declare more than one output (Split, and the fused
            # LSTM/GRU/RNN operators). Every declared output has to be registered,
            # otherwise downstream nodes cannot resolve their inputs.
            if isinstance(out, _SplitRequest):
                sizes = out.sizes
                if sizes is None:
                    # No "split" input supplied -> divide into equally sized parts.
                    quotient, remainder = divmod(out.total, len(n.output))
                    sizes = [quotient + (1 if i < remainder else 0) for i in range(len(n.output))]
                if len(sizes) != len(n.output):
                    raise Exception("Node '{}' (Split) declares {} outputs but {} split "
                                    "sizes were supplied.".format(n.name, len(n.output), len(sizes)))
                offset = 0
                for output_name, size in zip(n.output, sizes):
                    index = [slice(None), slice(None)]
                    index[out.axis] = slice(offset, offset + size)
                    value = ins[0][tuple(index)]
                    all_values[output_name] = value
                    node_values[output_name] = value
                    offset += size
            elif len(n.output) == 1:
                all_values[n.output[0]] = out
                node_values[n.output[0]] = out
            else:
                if not isinstance(out, (list, tuple)) or len(out) != len(n.output):
                    raise Exception("Operation '{}' declares {} outputs but returned {}."
                                    .format(n.op_type, len(n.output), type(out).__name__))
                for output_name, value in zip(n.output, out):
                    all_values[output_name] = value
                    node_values[output_name] = value
        
        # Assignment of all node output values to the class object 
        self.node_values = node_values
            
        

    def __getitem__(self, key: str):
        """ Enables the output of the CasADi expression of a specific layer or 
        graph operation node. 

        To learn about possible keywords, it is recommended to print the instance of the class:

        ::

            print(converter)

        Args:
            key: Name of the layer of the ONNX graph.


        """
        node_values = self.node_values

        if key in node_values.keys():
            out = node_values[key]
        else:
            raise Exception("The node '{}' is not contained in the ONNX graph.".format(key))

        return out



class ONNXOperations:
    """ CasADi operations, which are available in the :py:class:`ONNXConversion` class.
    See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_ for a full list of operations.

    .. note::

        This class is not intended to be used directly. It is used by the :py:class:`ONNXConversion` class.
        The purpose of this class is to provide a list of all available operations in the :py:class:`ONNXConversion` class.

    The currently implemented operators are:

    ==============  ==========================================================================================
    category        operators
    ==============  ==========================================================================================
    activation      ``Tanh``, ``Sigmoid``, ``Relu``, ``Elu``, ``LeakyRelu``, ``Softplus``, ``Softmax``,
                    ``Clip``
    arithmetic      ``Add``, ``Sub``, ``Mul``, ``Div``, ``Pow``, ``Sqrt``, ``Erf``, ``Abs``, ``Neg``,
                    ``Exp``, ``Log``, ``Sum``
    linear algebra  ``MatMul``, ``Gemm``
    normalisation   ``BatchNormalization`` (inference mode), ``LayerNormalization``
    reduction       ``ReduceMean``, ``ReduceSum``
    shape           ``Concat``, ``Split``, ``Transpose``, ``Unsqueeze``, ``Squeeze``, ``Slice``, ``Reshape``,
                    ``Flatten``, ``Shape``, ``Identity``
    constant        ``Constant``
    ==============  ==========================================================================================

    All of the above are verified against the ONNX specification and covered by
    ``testing/test_onnx.py``.

    .. note::

        **Smoothness matters more than support.** ``Relu``, ``LeakyRelu`` and
        ``Clip`` are implemented, but each has a kink where the derivative jumps.
        IPOPT is a gradient-based solver, so a network built from them can converge
        slowly or stall. When the network is going to be used as an MPC plant model,
        prefer ``Tanh``, ``Sigmoid``, ``Elu`` or ``Softplus`` -- and choose that at
        **training** time, since the activation cannot be swapped afterwards without
        retraining.

    Every method has the uniform signature ``(self, *args, attribute = None)``, where
    ``args`` are the node inputs in declaration order and ``attribute`` is the raw ONNX
    attribute list. Optional ONNX inputs are encoded as an empty string in the graph and
    are passed as ``None``, so an operator that accepts optional inputs must tolerate it.

    .. note::

        **Multi-output operators.** An operator that declares several outputs must return a
        ``list`` or ``tuple`` with one entry per declared output;
        :py:meth:`ONNXConversion.convert` registers them under the corresponding names.
        :py:meth:`Split` is special-cased: it returns a :py:class:`_SplitRequest` marker,
        because the number of equal-sized parts is only known from the node's output list.

    .. warning::

        **CasADi is strictly two-dimensional.** Operators that need a genuine rank-3
        (or higher) layout raise an explanatory exception instead of guessing:
        ``Transpose`` accepts only permutations that reduce to a 2-D transpose, and
        ``Concat`` accepts only axes that resolve to 0 or 1. ``Shape`` returns a plain
        Python tuple, so graphs that *compute with* shapes (``Shape`` -> ``Gather`` ->
        ``Concat`` -> ``Reshape``, the usual dynamic-reshape pattern) cannot be converted.

        The fused recurrent operators ``LSTM``, ``GRU`` and ``RNN`` are not implemented
        and cannot be, see :py:class:`ONNXConversion` for the ``Cell``-based workaround.
        Convolution and pooling (``Conv``, ``MaxPool``, ``AveragePool``), ``Gather``,
        ``Expand`` and all control-flow operators are not implemented. ``Conv`` in
        particular would need a three-dimensional sliding window; the practical
        workaround is to avoid convolutions in surrogate models, or to flatten them
        into an equivalent ``MatMul`` (im2col) before export.

    .. note::

        Both the opset <= 9 (attribute) and opset >= 10 (input) conventions are
        supported for ``Slice``, and ``Unsqueeze`` / ``Squeeze`` likewise read their
        axes from either an attribute or a second input. Every current exporter
        (``torch.onnx.export``, ``tf2onnx``) produces the opset >= 10 form.

    """
    def __init__(self):
        pass

    def Tanh(self,x, attribute = None):
        return casadi.tanh(x)

    def Sigmoid(self,x, attribute = None):
        out = 1/(1+casadi.exp(-x))
        return out

    def Relu(self,x, attribute = None):
        return casadi.fmax(0,x)

    def Elu(self,x, attribute = None):
        return casadi.fmax(0,x) + casadi.fmin(0,casadi.exp(x)-1)

    def LeakyRelu(self, x, attribute = None):
        """Leaky rectified linear unit.

        .. warning::
            Like :py:meth:`Relu` this has a kink at zero, so the derivative is
            discontinuous there. IPOPT is a gradient-based solver; prefer a
            smooth activation (``Tanh``, ``Sigmoid``, ``Elu``, ``Softplus``) when
            the network is going to be used inside an MPC.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#leakyrelu>`_ for more details.
        """
        alpha = self._attribute_float(attribute, 'alpha', 0.01)
        return casadi.fmax(0, x) + alpha * casadi.fmin(0, x)

    def Softplus(self, x, attribute = None):
        """Smooth approximation of :py:meth:`Relu`: ``log(1 + exp(x))``.

        Continuously differentiable, so unlike ``Relu`` / ``LeakyRelu`` it is safe
        to use inside an MPC.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#softplus>`_ for more details.
        """
        return casadi.log(1 + casadi.exp(x))

    def Softmax(self, x, attribute = None):
        """Normalised exponential along one axis.

        The row maximum is subtracted before exponentiating, which is the
        standard overflow-safe formulation and does not change the result.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#softmax>`_ for more details.
        """
        axis = self._resolve_axis(self._attribute_int(attribute, 'axis', -1))
        if axis == 1:
            # mmax / sum2 reduce across the columns and yield an (R, 1) result,
            # which CasADi broadcasts against (R, C) on its own.
            shifted = x - casadi.mmax(x)
            exponentiated = casadi.exp(shifted)
            return exponentiated / casadi.sum2(exponentiated)
        # mmax(x.T).T / sum1 reduce across the rows and yield a (1, C) result,
        # which has to be tiled explicitly.
        rows = int(x.shape[0])
        shifted = x - self._broadcast_rows(casadi.mmax(x.T).T, rows)
        exponentiated = casadi.exp(shifted)
        return exponentiated / self._broadcast_rows(casadi.sum1(exponentiated), rows)

    def Clip(self, x, minimum = None, maximum = None, attribute = None):
        """Clamp to ``[min, max]``.

        Both opset conventions are supported: from opset 11 the bounds are
        optional **inputs** (absent ones arrive as ``None``), before that they
        are attributes. ``torch.nn.Hardtanh`` and ``torch.clamp`` export here.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#clip>`_ for more details.
        """
        if minimum is None:
            minimum = self._attribute_float(attribute, 'min', None)
        if maximum is None:
            maximum = self._attribute_float(attribute, 'max', None)
        if minimum is not None:
            x = casadi.fmax(x, minimum)
        if maximum is not None:
            x = casadi.fmin(x, maximum)
        return x

    def MatMul(self,*args, attribute = None):
        return casadi.mtimes(*args)

    def Add(self,*args, attribute = None):
        """Addition of two or more tensors.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#add>`_ for more details.
        """
        out = 0
        for arg in args:
            out += arg
        return out

    def Mul(self,*args, attribute = None):
        return args[0]*args[1]

    def Sub(self, *args, attribute = None):
        return args[0] - args[1]

    def Div(self, *args, attribute = None):
        """Element-wise division.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#div>`_ for more details.
        """
        return args[0] / args[1]

    def Pow(self, *args, attribute = None):
        """Element-wise power.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#pow>`_ for more details.
        """
        return args[0] ** args[1]

    def Sqrt(self, x, attribute = None):
        """Element-wise square root.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#sqrt>`_ for more details.
        """
        return casadi.sqrt(x)

    def Erf(self, x, attribute = None):
        """Gauss error function. Together with ``Div``/``Mul``/``Add`` this is
        what ``torch.nn.GELU`` (the exact, erf-based variant) exports to.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#erf>`_ for more details.
        """
        return casadi.erf(x)

    def Abs(self, x, attribute = None):
        """Element-wise absolute value.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#abs>`_ for more details.
        """
        return casadi.fabs(x)

    def Neg(self, x, attribute = None):
        """Element-wise negation.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#neg>`_ for more details.
        """
        return -x

    def Exp(self, x, attribute = None):
        """Element-wise exponential.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#exp>`_ for more details.
        """
        return casadi.exp(x)

    def Log(self, x, attribute = None):
        """Element-wise natural logarithm.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#log>`_ for more details.
        """
        return casadi.log(x)

    def Gemm(self, *args, attribute = None):
        """General Matrix Multiplication.
        See `ONNX documentation  <https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm>`_ for more details.
        """

        attr_dict = {
            k.name: k.i if k.type == 2 else k.f for k in attribute
        }

        A = args[0]
        B = args[1]
        C = args[2]

        if 'transA' in attr_dict.keys() and attr_dict['transA'] == 1:
            A = casadi.transpose(A)
        if 'transB' in attr_dict.keys() and attr_dict['transB'] == 1:
            B = casadi.transpose(B)
        if 'alpha' in attr_dict.keys():
            alpha = attr_dict['alpha']
        else:
            alpha = 1
        if 'beta' in attr_dict.keys():
            beta = attr_dict['beta']
        else:
            beta = 1

        if C.ndim == 1:
            C = C.reshape(1,-1)
        
        res = alpha*self.MatMul(A,B) + beta*C

        return res

    def Sum(self,*args, attribute = None):
        return  self.Add(*args)

    def Concat(self, *args, attribute = None):
        """Concatenate tensors along one axis.

        The ``axis`` attribute defaults to ``0`` when absent, following the ONNX
        specification. Negative axes are resolved against CasADi's rank of two.

        Note:
            CasADi matrices are strictly two-dimensional, so only ``axis`` values
            that resolve to 0 (rows -> ``vertcat``) or 1 (columns -> ``horzcat``)
            are supported. Anything else raises, because the previous behaviour of
            silently mapping every other axis onto ``vertcat`` produced
            wrong-shaped results.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#concat>`_ for more details.
        """
        axis = self._attribute_int(attribute, 'axis', 0)
        try:
            resolved = self._resolve_axis(axis)
        except Exception:
            raise Exception("Concat with axis={} is not supported: it refers to a "
                            "dimension CasADi cannot hold.".format(axis))
        if resolved == 0:
            return casadi.vertcat(*args)
        return casadi.horzcat(*args)

    
    def Identity(self, x, attribute = None):
        """Return the input unchanged.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity>`_ for more details.
        """
        return x

    def Constant(self, *args, attribute = None):
        """A constant tensor. This operator takes **no** inputs; the value is
        carried by the ``value`` attribute (or one of its scalar variants).
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#constant>`_ for more details.
        """
        for attr in attribute or []:
            if attr.name != 'value':
                continue
            if attr.type == AttributeProto.TENSOR:
                return numpy_helper.to_array(attr.t)
            if attr.type == AttributeProto.INT:
                return np.array(attr.i)
            if attr.type == AttributeProto.FLOAT:
                return np.array(attr.f)
            if attr.type == AttributeProto.INTS:
                return np.array(list(attr.ints))
            if attr.type == AttributeProto.FLOATS:
                return np.array(list(attr.floats))
        raise Exception("Constant node without a supported 'value' attribute. "
                        "Only tensor, int, float, ints and floats values are implemented.")

    def Split(self, data, split_sizes = None, attribute = None):
        """Split a tensor into parts along one axis.

        This is a **multi-output** operator. Because the number of parts may be
        defined by the node's output list rather than by the ``split`` input, the
        actual slicing is performed by :py:meth:`ONNXConversion.convert`, which
        receives a :py:class:`_SplitRequest` marker from this method.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#split>`_ for more details.

        Args:
            data: Tensor to split.
            split_sizes: Optional size of every part. ``None`` requests an
                equal-sized split into as many parts as the node declares outputs.
            attribute: ONNX node attributes (used to read ``axis``).
        """
        axis = self._resolve_axis(self._attribute_int(attribute, 'axis', 0))
        sizes = None if split_sizes is None else [int(size) for size in np.ravel(split_sizes)]
        return _SplitRequest(axis, sizes, data.shape[axis])

    def Transpose(self, x, attribute = None):
        """Permute the axes of a tensor.

        Note:
            CasADi matrices are strictly two-dimensional. A genuine 3-D (or
            higher) permutation cannot be represented and raises an exception.
            Only ``perm`` values that reduce to a 2-D transpose or to a no-op
            are supported.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#transpose>`_ for more details.
        """
        perm = self._attribute_ints(attribute, 'perm')
        if perm is None:
            return casadi.transpose(x)
        offset = int(getattr(self, '_axis_offset_in', 0) or 0)
        onnx_rank = 2 + offset
        normalized = [int(p) % onnx_rank for p in perm]
        if sorted(normalized) != list(range(onnx_rank)):
            raise Exception("Transpose with perm={} is not a permutation of the "
                            "rank-{} tensor axes.".format(perm, onnx_rank))

        # Dimensions CasADi dropped must map onto dropped positions. A permutation
        # that leaves the leading singletons in place (e.g. [0, 2, 1] on a
        # [1, B, C] tensor) is perfectly representable; one that moves a dropped
        # dimension into the surviving pair is not.
        for position, source in enumerate(normalized):
            if (position < offset) != (source < offset):
                raise Exception("Transpose with perm={} is not supported: it moves a "
                                "dimension of size 1 that CasADi dropped into the "
                                "surviving two dimensions.".format(perm))

        casadi_perm = [source - offset for source in normalized if source >= offset]
        if casadi_perm == [0, 1]:
            return x
        if casadi_perm == [1, 0]:
            return casadi.transpose(x)
        raise Exception("Transpose with perm={} is not supported: CasADi is strictly "
                        "two-dimensional and cannot represent a genuine 3-D "
                        "permutation.".format(perm))

    def Unsqueeze(self, x, axes = None, attribute = None):
        """Insert axes of length one.

        Depending on the opset the axes are given either as an attribute
        (opset < 13) or as a second input (opset >= 13).

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#unsqueeze>`_ for more details.
        """
        requested = self._attribute_ints(attribute, 'axes')
        if requested is None and axes is not None:
            requested = [int(axis) for axis in np.ravel(axes)]

        # The axes refer to the RESULT tensor, whose ONNX rank is one higher per
        # inserted axis, so the output offset is the right frame of reference.
        offset = int(getattr(self, '_axis_offset_out', 0) or 0)
        onnx_rank = 2 + offset
        requested = sorted(int(axis) % onnx_rank for axis in (requested or []))
        shape = [1] * onnx_rank
        base = list(x.shape)
        # lay the two CasADi dimensions into the trailing ONNX positions
        shape[onnx_rank - 2:] = base
        for axis in requested:
            shape.insert(axis, 1)
        return self._to_casadi_shape(x, shape, len(base), 'Unsqueeze')

    def Squeeze(self, x, axes = None, attribute = None):
        """Remove axes of length one.

        Depending on the opset the axes are given either as an attribute
        (opset < 13) or as a second input (opset >= 13).

        Note:
            A resulting rank-1 tensor is reshaped into a column vector, since
            CasADi cannot represent a one-dimensional shape.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#squeeze>`_ for more details.
        """
        requested = self._attribute_ints(attribute, 'axes')
        if requested is None and axes is not None:
            requested = [int(axis) for axis in np.ravel(axes)]

        offset = int(getattr(self, '_axis_offset_in', 0) or 0)
        onnx_rank = 2 + offset
        shape = [1] * offset + list(x.shape)          # reconstruct the ONNX shape
        for axis in sorted((int(a) % onnx_rank for a in (requested or [])), reverse = True):
            if shape[axis] == 1:
                shape.pop(axis)
        return self._to_casadi_shape(x, shape, onnx_rank, 'Squeeze')

    @staticmethod
    def _reduce_to_2d(shape, operator = 'Reshape'):
        """Reduce an ONNX shape to the ``(rows, cols)`` pair CasADi can represent.

        Rules:

        * rank 0 -> ``(1, 1)``;
        * rank 1 -> ``(n, 1)``, a column vector;
        * rank 2 -> unchanged;
        * rank > 2 -> discard dimensions of size 1 (trailing ones first) until two
          remain. Discarding a singleton never changes the row-major element
          order, so this is lossless and covers the usual ``batch_size == 1``
          layout, e.g. ``(1, n, m)`` -> ``(n, m)``.

        Note:
            The **position** of the surviving dimensions matters. ``(1, 12)``
            reduces to the row vector ``(1, 12)`` while ``(12,)`` reduces to the
            column vector ``(12, 1)``. Collapsing every single-significant-dimension
            case onto a column vector would silently transpose row vectors.

        Note:
            **Dropping a dimension loses the original rank, and axis-referencing
            operators need it.** After a rank-3 tensor ``[1, B, C]`` is reduced to
            ``(B, C)``, ONNX ``axis=1`` refers to ``B`` -- the CasADi *rows* --
            while an operator that only sees the CasADi value would resolve
            ``axis=1`` to the *columns*. Negative axes are unaffected, because they
            count from the end, so ``axis=-1`` still means ``C``.

            This is why :py:class:`ONNXConversion` records the ONNX rank of every
            tensor (``_collect_ranks``, via :func:`onnx.shape_inference.infer_shapes`)
            and publishes the number of dropped leading dimensions as
            ``_axis_offset_in`` / ``_axis_offset_out`` before each node is
            dispatched. Operators pass their axes through
            :py:meth:`ONNXOperations._resolve_axis`, which remaps a positive axis
            onto the surviving pair, leaves a negative axis alone and rejects an
            axis that is out of range or that addresses a dropped dimension.

            A genuine rank-3 graph *input* still cannot be converted: CasADi has no
            rank-3 type, and ``casadi.SX.sym(name, 1, 1, 3)`` returns a ``list``,
            not an ``SX``. The window is therefore limited to a tensor that was
            rank 2 on entry and raised to rank 3 by ``Unsqueeze`` / ``Reshape``
            before an axis operator consumes it.

        Args:
            shape: Target ONNX shape.
            operator: Operator name, used in the error message.

        Returns:
            tuple: ``(rows, cols)``.

        Raises:
            Exception: if more than two non-singleton dimensions remain.
        """
        reduced = [int(dimension) for dimension in shape]
        while len(reduced) > 2:
            index = None
            for position in range(len(reduced) - 1, -1, -1):
                if reduced[position] == 1:
                    index = position
                    break
            if index is None:
                raise Exception(
                    "{} cannot produce the shape {}: CasADi is strictly "
                    "two-dimensional and no dimension of size 1 can be dropped "
                    "without losing data.".format(operator, tuple(reduced)))
            reduced.pop(index)
        if len(reduced) == 2:
            return reduced[0], reduced[1]
        if len(reduced) == 1:
            return reduced[0], 1
        return 1, 1

    @staticmethod
    def _to_casadi_shape(x, shape, original_rank, operator):
        """Reshape ``x`` to ``shape``, working around CasADi's strict 2-D layout.

        CasADi matrices always have exactly two dimensions, while ONNX tensors may
        have any rank. This helper maps an ONNX shape onto a CasADi shape:

        * rank 2 is used as-is;
        * rank 1 becomes a column vector ``(n, 1)``;
        * rank 0 becomes ``(1, 1)``;
        * rank > 2 is reduced by discarding dimensions of size 1 until two
          dimensions remain. Discarding a singleton never changes the element
          order of a CasADi matrix, so this is lossless and covers the usual
          ``batch_size == 1`` situation. If the rank is still above 2 afterwards
          an exception is raised, because dropping a dimension of size > 1 would
          silently change the semantics.

        Args:
            x: Tensor to reshape.
            shape: Target ONNX shape.
            original_rank: Rank of ``x`` before the operation (for the error message).
            operator: Operator name (for the error message).

        Raises:
            Exception: if the target shape cannot be represented without losing data.
        """
        rows, cols = ONNXOperations._reduce_to_2d(shape, operator)
        expected = rows * cols
        actual = int(x.size) if isinstance(x, np.ndarray) else int(x.shape[0]) * int(x.shape[1])
        if expected != actual:
            raise Exception(
                "{} cannot reshape {} elements into the shape {}: the number of "
                "elements does not match.".format(operator, actual, tuple(shape)))
        # Note: the bound method form only accepts a single tuple argument.
        return x.reshape((rows, cols))

    def _resolve_axis(self, axis, which = 'in'):
        """Map an ONNX axis onto the CasADi axis it actually refers to.

        ``ONNXConversion.convert`` records how many leading singleton dimensions
        were dropped when the tensor was stored in CasADi (see
        ``_axis_offset_in`` / ``_axis_offset_out``). Without that offset a
        **positive** axis addresses the wrong dimension: for an ONNX tensor
        ``[1, B, C]`` held as ``(B, C)``, ``axis=1`` is the ``B`` axis, i.e. the
        CasADi *rows*, not the columns.

        Negative axes need no correction -- they count from the end, and only
        leading dimensions are ever dropped -- but they are normalised through the
        ONNX rank so that an axis referring to a dropped dimension is detected
        rather than silently wrapping around.

        Args:
            axis: The axis as written in the ONNX graph.
            which: ``'in'`` for operators whose axes refer to the input tensor
                (most of them, and ``Squeeze``), ``'out'`` for ``Unsqueeze``,
                whose axes refer to the result.

        Returns:
            int: ``0`` (CasADi rows) or ``1`` (CasADi columns).

        Raises:
            Exception: if the axis refers to a dimension that CasADi cannot hold.
        """
        attribute_name = '_axis_offset_in' if which == 'in' else '_axis_offset_out'
        offset = int(getattr(self, attribute_name, 0) or 0)
        onnx_rank = 2 + offset
        axis = int(axis)
        if not -onnx_rank <= axis < onnx_rank:
            # Without this check a modulo would silently wrap axis=2 onto axis=0
            # for a rank-2 tensor, which is an invalid graph rather than a no-op.
            raise Exception("axis {} is out of range for a rank-{} tensor."
                            .format(axis, onnx_rank))
        normalized = axis % onnx_rank
        casadi_axis = normalized - offset
        if casadi_axis < 0:
            raise Exception("axis {} refers to a dimension of size 1 that CasADi "
                            "dropped when reducing the rank-{} tensor to two "
                            "dimensions, so it cannot be addressed."
                            .format(axis, onnx_rank))
        return casadi_axis

    @staticmethod
    def _attribute_int(attribute, name, default):
        """Read an integer attribute, returning ``default`` if it is absent."""
        for attr in attribute or []:
            if attr.name == name and attr.type == AttributeProto.INT:
                return int(attr.i)
        return default

    @staticmethod
    def _attribute_ints(attribute, name):
        """Read a list-of-ints attribute, returning ``None`` if it is absent."""
        for attr in attribute or []:
            if attr.name == name and attr.type == AttributeProto.INTS:
                return [int(value) for value in attr.ints]
        return None

    @staticmethod
    def _attribute_float(attribute, name, default):
        """Read a float attribute, returning ``default`` if it is absent."""
        for attr in attribute or []:
            if attr.name == name and attr.type == AttributeProto.FLOAT:
                return float(attr.f)
        return default

    @staticmethod
    def _broadcast_rows(row, rows):
        """Repeat a ``(1, C)`` row vector down to ``(rows, C)``.

        CasADi only broadcasts a scalar or a ``(R, 1)`` column against an
        ``(R, C)`` matrix. Operating with a ``(1, C)`` row raises
        ``Dimension mismatch for (x-y), x is RxC, while y is 1xC``, so per-channel
        parameters have to be tiled explicitly.
        """
        rows = int(rows)
        if rows == 1:
            return row
        return casadi.repmat(row, rows, 1)

    @staticmethod
    def _as_row(vector):
        """Reshape a per-channel parameter to a row vector ``(1, C)``.

        ONNX stores the scale / bias / mean / variance of normalisation layers as
        one-dimensional tensors of shape ``(C,)``. CasADi has no rank-1 type, and
        broadcasting a ``(C,)`` numpy array against a ``(1, C)`` CasADi row raises
        a dimension mismatch, so the parameters are reshaped explicitly.
        """
        if isinstance(vector, np.ndarray):
            return np.ravel(vector).reshape(1, -1)
        return casadi.reshape(vector, (1, vector.shape[0]))

    @staticmethod
    def _as_matrix(value, rows, cols):
        """Interpret a parameter tensor as a CasADi ``(rows, cols)`` matrix.

        The scale / bias of a ``LayerNormalization`` whose normalised group spans
        more than one dimension have the shape of that group, i.e. the full
        ``(rows, cols)`` matrix. They must **not** be raveled the way
        :py:meth:`_as_row` ravel's per-channel parameters: CasADi stores matrices
        column-major while ONNX tensors are row-major, so flattening would
        scramble the element order.

        Args:
            value: Parameter tensor, a numpy array or a CasADi expression.
            rows: Number of rows of the matrix it belongs to.
            cols: Number of columns of the matrix it belongs to.

        Returns:
            A ``(rows, cols)`` CasADi expression aligned with ``x``.
        """
        if isinstance(value, np.ndarray):
            return casadi.DM(np.asarray(value).reshape((rows, cols)))
        return casadi.reshape(value, (rows, cols))

    def _reduce_axis(self, x, axes_input, attribute, default):
        """Resolve the axis of a Reduce* operator.

        ``axes`` moved from an attribute to an optional input in opset 18, so
        both forms are accepted. The result is normalised against CasADi's rank
        of two.
        """
        axes = None
        for attr in attribute or []:
            if attr.name == 'axes' and attr.type == AttributeProto.INTS:
                axes = [int(value) for value in attr.ints]
            if attr.name == 'axis' and attr.type == AttributeProto.INT:
                axes = [int(attr.i)]
        if axes is None and axes_input is not None:
            axes = [int(value) for value in np.ravel(axes_input)]
        if not axes:
            axes = [default]
        if len(axes) != 1:
            raise Exception("Reducing over several axes at once ({}) is not "
                            "supported: CasADi is strictly two-dimensional."
                            .format(axes))
        return self._resolve_axis(axes[0])

    def Slice(self, data, starts = None, ends = None, axes = None, steps = None,
              attribute = None):
        """Extract a sub-tensor.

        Both ONNX conventions are supported:

        * **opset >= 10** (what every current exporter produces): ``starts``,
          ``ends``, ``axes`` and ``steps`` are regular **inputs**, any of which may
          be omitted as an empty string and is then passed as ``None``.
        * **opset <= 9**: the same values are **attributes**.

        The ONNX clamping rules are implemented: negative indices count from the
        end, ``INT_MAX`` means "to the end", ``INT_MIN`` means "from the
        beginning", and out-of-range values are clamped to the dimension size.

        Note:
            CasADi matrices are strictly two-dimensional, so ``data`` must have
            rank <= 2 and every entry of ``axes`` must resolve to 0 or 1.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice>`_ for more details.

        Args:
            data: Tensor to slice.
            starts: Starting index per axis (input or, for opset <= 9, attribute).
            ends: Ending index per axis (exclusive).
            axes: Axes the indices refer to. Defaults to ``range(len(starts))``.
            steps: Step size per axis. Defaults to all ones.
            attribute: ONNX node attributes (opset <= 9 form).

        Raises:
            Exception: if starts/ends are missing, a step is zero, or the tensor
                rank / axis cannot be represented in CasADi.
        """
        if starts is None or ends is None:
            attrs = {attr.name: list(attr.ints) for attr in (attribute or [])}
            starts = attrs.get('starts') if starts is None else starts
            ends = attrs.get('ends') if ends is None else ends
            axes = attrs.get('axes') if axes is None else axes
            steps = attrs.get('steps') if steps is None else steps
        if starts is None or ends is None:
            raise Exception("Slice requires 'starts' and 'ends', either as inputs "
                            "(opset >= 10) or as attributes (opset <= 9).")

        starts = [int(v) for v in np.ravel(starts)]
        ends = [int(v) for v in np.ravel(ends)]
        rank = len(data.shape)
        if rank > 2:
            raise Exception("Slice cannot handle a rank-{} tensor: CasADi is strictly "
                            "two-dimensional.".format(rank))
        if axes is None:
            axes = list(range(len(starts)))
        else:
            axes = [int(v) for v in np.ravel(axes)]
        steps = [1] * len(starts) if steps is None else [int(v) for v in np.ravel(steps)]
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise Exception("Slice got inconsistent lengths: starts={}, ends={}, axes={}, "
                            "steps={}".format(len(starts), len(ends), len(axes), len(steps)))

        int_max = int(np.iinfo(np.int64).max)
        int_min = int(np.iinfo(np.int64).min)
        index = [slice(None, None)] * rank

        for axis, start, end, step in zip(axes, starts, ends, steps):
            axis = self._resolve_axis(axis)
            if step == 0:
                raise Exception("Slice: 'steps' must not be zero (axis {}).".format(axis))
            dim = int(data.shape[axis])

            if step > 0:
                # INT_MIN / out-of-range low  -> beginning; INT_MAX / high -> end
                start = 0 if start <= int_min or start < -dim else start
                start = dim if start >= int_max else start
                start = start + dim if start < 0 else start
                end = dim if end >= int_max or end > dim or end < -dim else end
                end = end + dim if end < 0 else end
                start = min(max(start, 0), dim)
                end = min(max(end, 0), dim)
            else:
                # Reversed traversal: start defaults to the last element and end
                # may be INT_MIN, meaning "one before the first element".
                start = dim - 1 if start >= int_max else start
                start = start + dim if -dim <= start < 0 else start
                start = min(max(start, -1), dim - 1)
                end = -dim - 1 if end <= int_min or end < -dim - 1 else end
                end = end + dim if -dim <= end < 0 else end
                end = max(min(end, dim), -dim - 1)

            index[axis] = slice(start, end, None if step == 1 else step)

        return data[tuple(index)]

    def Reshape(self, data, shape = None, attribute = None):
        """Change the shape of a tensor without changing its data.

        The ONNX special values are honoured: ``0`` copies the corresponding
        dimension from the input and ``-1`` infers it from the element count.

        Note:
            **Element order follows ONNX, i.e. row-major.** CasADi's own
            ``reshape`` is column-major, so a plain ``data.reshape(target)`` would
            silently permute the values. This implementation flattens ``data`` in
            row-major order and rebuilds the target shape from that.

        Note:
            CasADi is strictly two-dimensional. Target shapes whose
            non-singleton dimensions number more than two cannot be represented
            and raise an exception.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#reshape>`_ for more details.

        Args:
            data: Tensor to reshape.
            shape: Target shape, as the node's second input.
            attribute: Unused; present for signature uniformity.

        Raises:
            Exception: if ``shape`` is missing, the element counts disagree, or
                the target rank cannot be represented in CasADi.
        """
        if shape is None:
            raise Exception("Reshape requires the target shape as its second input.")
        if isinstance(data, np.ndarray):
            in_shape = tuple(int(dimension) for dimension in data.shape)
        else:
            in_shape = (int(data.shape[0]), int(data.shape[1]))
        numel = 1
        for dimension in in_shape:
            numel *= dimension

        target = [int(value) for value in np.ravel(shape)]
        resolved = []
        for position, value in enumerate(target):
            if value == 0:
                resolved.append(in_shape[position] if position < len(in_shape) else 1)
            else:
                resolved.append(value)
        if -1 in resolved:
            known = 1
            for value in resolved:
                if value != -1:
                    known *= value
            if known == 0 or numel % known != 0:
                raise Exception("Reshape cannot infer the -1 dimension: {} elements do "
                                "not divide evenly by {}.".format(numel, known))
            resolved[resolved.index(-1)] = numel // known

        product = 1
        for dimension in resolved:
            product *= dimension
        if product != numel:
            raise Exception("Reshape cannot reshape {} elements into the shape {}: the "
                            "number of elements does not match.".format(numel, tuple(resolved)))

        # Resolve the rank first so the error message names Reshape.
        rows, cols = self._reduce_to_2d(resolved, 'Reshape')

        if isinstance(data, np.ndarray):
            # numpy is row-major natively and dropping singletons preserves order
            return np.reshape(data, tuple(resolved)).reshape(rows, cols)

        # CasADi is column-major while ONNX Reshape is row-major, so a plain
        # data.reshape(target) would silently permute the values. Flatten in
        # row-major order and rebuild: for a target of (rows, cols) the flat
        # vector is laid out as a (cols, rows) column-major matrix and transposed.
        flat = casadi.vec(casadi.transpose(data))
        return casadi.transpose(casadi.reshape(flat, (cols, rows)))

    def Shape(self, x, attribute = None):
        """Return the shape of a tensor.

        Warning:
            This returns a plain Python **tuple**, not a CasADi value, because a
            shape is meta-information rather than a differentiable quantity. It is
            therefore only useful for concrete (numeric) tensors. A symbolic graph
            that computes with shapes -- e.g. ``Shape`` -> ``Gather`` -> ``Concat``
            -> ``Reshape``, which is how exporters build dynamic reshapes -- cannot
            be represented, and will fail on the first operator that tries to use
            the tuple as a CasADi value.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#shape>`_ for more details.
        """
        return tuple(x.shape)

    def Flatten(self, x, attribute = None):
        """Flatten into a two-dimensional tensor.

        ``axis`` (default 1) splits the input shape into the row and the column
        part. Since CasADi is already two-dimensional this is a no-op for
        ``axis == 1`` and a full flatten for the other values.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten>`_ for more details.
        """
        axis = self._resolve_axis(self._attribute_int(attribute, 'axis', 1))
        rows, cols = int(x.shape[0]), int(x.shape[1])
        total = rows * cols
        # ONNX: the output is (d0*...*d_{axis-1}, d_axis*...*d_n). For a rank-2
        # CasADi matrix that gives axis 0 -> (1, R*C), axis 1 -> (R, C) unchanged
        # and axis 2 -> (R*C, 1).
        if axis == 1:
            return x
        target = (1, total) if axis <= 0 else (total, 1)
        # Delegate to Reshape: a plain x.reshape(target) would follow CasADi's
        # column-major order, while ONNX Flatten is row-major.
        return self.Reshape(x, np.array(target, dtype=np.int64))

    def BatchNormalization(self, x, scale, bias, mean, var, attribute = None):
        """Batch normalisation in **inference** mode.

        The running mean and variance are initializers, so this reduces to an
        affine map and needs no state:

        .. math::

            y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta

        Note:
            Only the default ``training_mode = 0`` is supported. Training-mode
            graphs would need the batch statistics, which are not available at
            conversion time.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization>`_ for more details.
        """
        if self._attribute_int(attribute, 'training_mode', 0) != 0:
            raise Exception('BatchNormalization in training mode is not supported; '
                            'export the model in eval mode so the running statistics '
                            'are baked in as initializers.')
        epsilon = self._attribute_float(attribute, 'epsilon', 1e-5)
        rows = int(x.shape[0])
        scale, bias, mean, var = (self._broadcast_rows(self._as_row(value), rows)
                                  for value in (scale, bias, mean, var))
        return (x - mean) / casadi.sqrt(var + epsilon) * scale + bias

    def LayerNormalization(self, x, scale, bias = None, attribute = None):
        """Layer normalisation.

        .. math::

            y = \\frac{x - \\mathrm{mean}(x)}{\\sqrt{\\mathrm{var}(x) + \\epsilon}} \\cdot \\gamma + \\beta

        Note:
            ONNX normalises over **all** dimensions from ``axis`` to the last one,
            and ``scale`` / ``bias`` have the shape of that whole group. When the
            axis maps onto the CasADi columns (``axis=-1`` or ``axis=rank-1``, by
            far the common case) the group is that single axis. When it maps onto
            the rows the group covers the whole matrix and is handled by
            :py:meth:`_layer_norm_over_all_elements`.

        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#layernormalization>`_ for more details.
        """
        epsilon = self._attribute_float(attribute, 'epsilon', 1e-5)
        axis = self._resolve_axis(self._attribute_int(attribute, 'axis', -1))
        if axis == 0:
            # ONNX normalises over ``normalized_axes = [axis, ..., rank - 1]``
            # *jointly*, so the group always reaches the last dimension. Once the
            # axis maps onto the CasADi rows it therefore spans BOTH CasADi axes,
            # whatever the ONNX rank is: ``axis=0`` on a rank-2 tensor and
            # ``axis=1`` on a rank-3 ``[1, B, C]`` tensor are the same case.
            # Reducing a single axis here would silently return a mis-shaped,
            # wrong result.
            return self._layer_norm_over_all_elements(x, scale, bias, epsilon)

        # _resolve_axis only ever returns 0 or 1, so this is the columns case:
        # the group is the last ONNX dimension alone. (R, 1) reductions broadcast
        # against (R, C) natively.
        rows, count = int(x.shape[0]), int(x.shape[1])
        mean = casadi.sum2(x) / count
        centred = x - mean
        variance = casadi.sum2(centred * centred) / count
        scale = self._broadcast_rows(self._as_row(scale), rows)
        out = centred / casadi.sqrt(variance + epsilon) * scale
        if bias is not None:
            out = out + self._broadcast_rows(self._as_row(bias), rows)
        return out

    def _layer_norm_over_all_elements(self, x, scale, bias, epsilon):
        """Layer normalisation whose group covers the whole CasADi matrix.

        ONNX defines ``normalized_axes = [axis, ..., rank - 1]``, so the group
        reaches the last dimension and spans both CasADi axes as soon as the
        resolved axis is ``0``. Typical cases are ``LayerNormalization(axis=0)``
        on a rank-2 tensor (``torch.nn.LayerNorm([rows, cols])``) and
        ``LayerNormalization(axis=1)`` on a ``[1, B, C]`` tensor held as
        ``(B, C)``.

        Only the mean and the variance are reductions, and a total sum is
        independent of the element order, so the row-major / column-major
        difference between ONNX and CasADi does not matter here. The per-element
        scale and bias keep the ``(rows, cols)`` layout (see
        :py:meth:`_as_matrix`).

        Args:
            x: Tensor to normalise, a ``(rows, cols)`` CasADi expression.
            scale: Gain of the normalisation group.
            bias: Offset of the normalisation group, may be ``None``.
            epsilon: Term added to the variance for numerical stability.

        Returns:
            The normalised, scaled and shifted tensor.
        """
        rows, cols = int(x.shape[0]), int(x.shape[1])
        count = rows * cols
        mean = casadi.sum1(casadi.sum2(x)) / count
        centred = x - mean
        variance = casadi.sum1(casadi.sum2(centred * centred)) / count
        out = centred / casadi.sqrt(variance + epsilon) * self._as_matrix(scale, rows, cols)
        if bias is not None:
            out = out + self._as_matrix(bias, rows, cols)
        return out

    def ReduceMean(self, x, axes = None, attribute = None):
        """Mean along one axis.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#reducemean>`_ for more details.
        """
        axis = self._reduce_axis(x, axes, attribute, default = -1)
        total = casadi.sum2(x) if axis == 1 else casadi.sum1(x)
        return total / int(x.shape[axis])

    def ReduceSum(self, x, axes = None, attribute = None):
        """Sum along one axis.
        See `ONNX documentation <https://github.com/onnx/onnx/blob/main/docs/Operators.md#reducesum>`_ for more details.
        """
        axis = self._reduce_axis(x, axes, attribute, default = -1)
        return casadi.sum2(x) if axis == 1 else casadi.sum1(x)


    

