"""
Model Utilities Module

This module contains model replacement and transformation functions:
- CIM layer wrappers (SplitQKV_CIM_Wrapper)
- ViT layer replacement for CIM
- Fake quant module wrapping
"""

import torch
import torch.nn as nn

from .CIM_Quant import CIM_Linear, CIM_SM_Linear
from .Fake_Quant import NoisyLinear, NoisyConv2d


def get_cim_layer_class(weight_encode_method):
    """
    Select the CIM layer class according to weight encoding mode.

    Args:
        weight_encode_method: 'twos_complement' or 'differential'

    Returns:
        CIM layer class (CIM_Linear or CIM_SM_Linear)
    """
    if weight_encode_method == 'twos_complement':
        return CIM_Linear
    return CIM_SM_Linear


class SplitQKV_CIM_Wrapper(nn.Module):
    """
    Replace a fused ViT qkv Linear layer with three independent CIM layers.

    This keeps the original qkv behavior but allows each branch to use
    CIM_Linear or CIM_SM_Linear.

    Args:
        original_linear: The original fused QKV linear layer
        parallel_read: Number of rows for parallel read
        variation_sigma: Variation sigma value
        adc_bits: ADC bit resolution
        input_bits: Input quantization bits
        weight_bits: Weight quantization bits
        use_partial_sum_quant: Whether to use partial sum quantization
        weight_encode_method: 'twos_complement' or 'differential' (for weights)
        activation_encode_method: 'twos_complement' or 'differential' (for activations)
        enable_activity_stats: Whether to enable activity counting
    """
    def __init__(self, original_linear, parallel_read, variation_sigma, adc_bits,
                 input_bits, weight_bits, use_partial_sum_quant,
                 weight_encode_method, activation_encode_method,
                 enable_activity_stats=False, enable_sparsity_stats=False):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        head_dim = out_features // 3

        assert out_features % 3 == 0, "QKV output features must be divisible by 3"

        # Select CIM layer class based on weight encoding method
        cim_cls = CIM_Linear if weight_encode_method == 'twos_complement' else CIM_SM_Linear

        common_kwargs = {
            'input_bits': input_bits,
            'weight_bits': weight_bits,
            'adc_bits': adc_bits,
            'rows_parallel': parallel_read,
            'variation_sigma': variation_sigma,
            'bias': original_linear.bias is not None,
            'use_partial_sum_quant': use_partial_sum_quant,
            'activation_encode_method': activation_encode_method,
            'enable_activity_stats': enable_activity_stats,
            'enable_sparsity_stats': enable_sparsity_stats,
        }

        self.q = cim_cls(in_features, head_dim, **common_kwargs)
        self.k = cim_cls(in_features, head_dim, **common_kwargs)
        self.v = cim_cls(in_features, head_dim, **common_kwargs)

        w = original_linear.weight.data
        self.q.weight.data = w[0:head_dim, :].clone()
        self.k.weight.data = w[head_dim:2 * head_dim, :].clone()
        self.v.weight.data = w[2 * head_dim:, :].clone()

        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.q.bias.data = b[0:head_dim].clone()
            self.k.bias.data = b[head_dim:2 * head_dim].clone()
            self.v.bias.data = b[2 * head_dim:].clone()

    def forward(self, x):
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        return torch.cat([q_out, k_out, v_out], dim=-1)


def replace_vit_layers_with_cim(model, parallel_read, variation_sigma, adc_bits,
                                  input_bits, weight_bits, use_partial_sum_quant,
                                  weight_encode_method, activation_encode_method,
                                  enable_activity_stats=False, enable_sparsity_stats=False):
    """
    Replace ViT Linear layers with CIM-based layers.

    Replaced modules:
    - block.attn.qkv
    - block.attn.proj
    - block.mlp.fc1
    - block.mlp.fc2

    Args:
        model: The ViT model to modify
        parallel_read: Number of rows for parallel read
        variation_sigma: Variation sigma value
        adc_bits: ADC bit resolution
        input_bits: Input quantization bits
        weight_bits: Weight quantization bits
        use_partial_sum_quant: Whether to use partial sum quantization
        weight_encode_method: 'twos_complement' or 'differential' (for weights)
        activation_encode_method: 'twos_complement' or 'differential' (for activations)
        enable_activity_stats: Whether to enable activity counting

    Returns:
        Modified model with CIM layers
    """
    cim_layer_cls = get_cim_layer_class(weight_encode_method)

    common_kwargs = {
        'input_bits': input_bits,
        'weight_bits': weight_bits,
        'adc_bits': adc_bits,
        'rows_parallel': parallel_read,
        'variation_sigma': variation_sigma,
        'use_partial_sum_quant': use_partial_sum_quant,
        'activation_encode_method': activation_encode_method,
        'enable_activity_stats': enable_activity_stats,
        'enable_sparsity_stats': enable_sparsity_stats,
    }

    block_idx = 0
    for block in model.blocks:
        # Replace QKV
        if hasattr(block.attn, 'qkv'):
            original_qkv = block.attn.qkv
            new_qkv = SplitQKV_CIM_Wrapper(
                original_qkv, parallel_read, variation_sigma, adc_bits,
                input_bits, weight_bits, use_partial_sum_quant,
                weight_encode_method, activation_encode_method,
                enable_activity_stats, enable_sparsity_stats
            )
            # Add layer names to sub-modules (q, k, v)
            new_qkv.q.layer_name = f'block{block_idx}.attn.q'
            new_qkv.k.layer_name = f'block{block_idx}.attn.k'
            new_qkv.v.layer_name = f'block{block_idx}.attn.v'
            block.attn.qkv = new_qkv

        # Replace projection
        if hasattr(block.attn, 'proj'):
            original_proj = block.attn.proj
            new_proj = cim_layer_cls(
                original_proj.in_features,
                original_proj.out_features,
                bias=original_proj.bias is not None,
                **common_kwargs
            )
            new_proj.layer_name = f'block{block_idx}.attn.proj'
            new_proj.weight.data = original_proj.weight.data.clone()
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data.clone()
            block.attn.proj = new_proj

        # Replace MLP layers
        if hasattr(block, 'mlp'):
            for fc_name in ['fc1', 'fc2']:
                if hasattr(block.mlp, fc_name):
                    original_fc = getattr(block.mlp, fc_name)
                    new_fc = cim_layer_cls(
                        original_fc.in_features,
                        original_fc.out_features,
                        bias=original_fc.bias is not None,
                        **common_kwargs
                    )
                    new_fc.layer_name = f'block{block_idx}.mlp.{fc_name}'
                    new_fc.weight.data = original_fc.weight.data.clone()
                    if original_fc.bias is not None:
                        new_fc.bias.data = original_fc.bias.data.clone()
                    setattr(block.mlp, fc_name, new_fc)

        block_idx += 1

    return model


def wrap_fake_quant_modules(model, sigma, weight_bits, input_bits,
                            weight_encode_method, activation_encode_method,
                            noise_enable, noise_mode, include_layers,
                            exclude_layers):
    """
    Recursively replace Linear/Conv2d with Fake_quant wrapper layers.

    This is the simplified quant/noise path.

    Args:
        model: The model to modify
        sigma: Variation sigma value
        weight_bits: Weight quantization bits
        input_bits: Input quantization bits
        weight_encode_method: 'twos_complement' or 'differential' (for weights)
        activation_encode_method: 'twos_complement' or 'differential' (for activations)
        noise_enable: Whether to enable noise
        noise_mode: 'include' or 'exclude'
        include_layers: Layers to include for noise (if mode is include)
        exclude_layers: Layers to exclude from noise (if mode is exclude)

    Returns:
        Modified model with Fake quant layers
    """
    def _wrap(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                new = NoisyLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    layer_name=full_name,
                    num_bits_weight=weight_bits,
                    num_bits_act=input_bits,
                    encoding_mode=weight_encode_method,
                    activation_encoding_mode=activation_encode_method,
                    noise_enable=noise_enable,
                    noise_sigma=sigma,
                    noise_mode=noise_mode,
                    include_layers=include_layers,
                    exclude_layers=exclude_layers
                )
                new.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.data)
                setattr(module, name, new)

            elif isinstance(child, nn.Conv2d):
                new = NoisyConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    layer_name=full_name,
                    num_bits_weight=weight_bits,
                    num_bits_act=input_bits,
                    encoding_mode=weight_encode_method,
                    activation_encoding_mode=activation_encode_method,
                    noise_enable=noise_enable,
                    noise_sigma=sigma,
                    noise_mode=noise_mode,
                    include_layers=include_layers,
                    exclude_layers=exclude_layers
                )
                new.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.data)
                setattr(module, name, new)

            else:
                _wrap(child, full_name)

    _wrap(model)
    return model