"""
Tensor Networks for Finance
A Python package for applying tensor network methods to financial applications
"""

__version__ = "0.1.0"
__author__ = "Ian Hincks"

from .tensor_networks import (
    BasketOptionTN,
    TensorLMM,
    TensorLayer,
    compare_layer_sizes,
    cholesky_correlation_transform,
    independent_to_correlated
)

__all__ = [
    'BasketOptionTN',
    'TensorLMM',
    'TensorLayer',
    'compare_layer_sizes',
    'cholesky_correlation_transform',
    'independent_to_correlated'
]
