"""
Quantization Layers

This module re-exports CIM and Fake Quant layers for backward compatibility.

Import from:
- CIM_Quant: CIM_Linear, CIM_SM_Linear
- Fake_Quant: NoisyLinear, NoisyConv2d
"""

from .CIM_Quant import CIM_Linear, CIM_SM_Linear
from .Fake_Quant import NoisyLinear, NoisyConv2d

__all__ = [
    'CIM_Linear',
    'CIM_SM_Linear',
    'NoisyLinear',
    'NoisyConv2d',
]