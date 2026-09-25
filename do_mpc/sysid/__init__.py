"""
Tools for machine learning and system identification.

.. warning::
    The :py:class:`ONNXConversion` class is  experimental.
    
"""

import warnings
from .. import __ONNX_INSTALLED__

if __ONNX_INSTALLED__:
    from ._onnxconversion import ONNXConversion, ONNXOperations
    from ._anntomodel import (ann_to_dompc_model, torch_module_to_onnx, validate_wiring,
                            set_constant_parameters, set_constant_tvp, resolve_output)
else:
    warnings.warn('The ONNX feature is not available. Please install the full version of do-mpc to access this feature.')