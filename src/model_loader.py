"""
Model Loader Module

This module contains model construction and weight loading functions:
- Build empty model
- Load checkpoint
- Load state dict
- Build model for evaluation
"""

import os
import timm
import torch
import torch.nn as nn


def build_empty_model(model_name='vit_tiny_patch16_224', num_classes=100):
    """
    Create the base ViT model structure.

    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes

    Returns:
        Model instance
    """
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def load_checkpoint_to_model(model, checkpoint_path):
    """
    Load a checkpoint file into the given model.

    Supported checkpoint formats:
    - checkpoint['model_state_dict']
    - checkpoint['state_dict']
    - checkpoint['model']
    - raw state_dict

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file

    Returns:
        Model with loaded weights

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Path not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def load_state_dict_to_model(model, state_dict):
    """
    Load an in-memory state_dict into the given model.

    Args:
        model: The model to load weights into
        state_dict: The state dict to load

    Returns:
        Model with loaded weights
    """
    model.load_state_dict(state_dict, strict=True)
    return model


def load_model_from_checkpoint(checkpoint_path, model_name='vit_tiny_patch16_224', num_classes=100):
    """
    Build a fresh base model and load weights from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Name of the model to create
        num_classes: Number of output classes

    Returns:
        Model with loaded weights
    """
    model = build_empty_model(model_name, num_classes)
    model = load_checkpoint_to_model(model, checkpoint_path)
    return model


def build_model_for_current_mode(sigma, base_state_dict, quant_mode, weight_encode_method,
                                  activation_encode_method, weight_bits, input_bits,
                                  adc_bits, parallel_read, use_partial_sum_quant,
                                  noise_enable, noise_mode, include_layers,
                                  exclude_layers, device, num_classes=100,
                                  enable_activity_stats=False,
                                  enable_sparsity_stats=False):
    """
    Build one evaluation model for the current sigma value.

    Order:
    1. rebuild base model
    2. load in-memory pruned/non-pruned state_dict
    3. apply quant-layer replacement according to quant_mode
    4. move to DEVICE and set eval mode

    Args:
        sigma: Variation sigma value
        base_state_dict: Base state dict (pruned or original)
        quant_mode: 'CIM_Quant' or 'Fake_Quant'
        weight_encode_method: 'twos_complement' or 'differential' (for weights)
        activation_encode_method: 'twos_complement' or 'differential' (for activations)
        weight_bits: Weight quantization bits
        input_bits: Input quantization bits
        adc_bits: ADC bit resolution
        parallel_read: Number of rows for parallel read
        use_partial_sum_quant: Whether to use partial sum quantization
        noise_enable: Whether to enable noise
        noise_mode: 'include' or 'exclude'
        include_layers: Layers to include for noise
        exclude_layers: Layers to exclude from noise
        device: Device to move model to
        num_classes: Number of output classes

    Returns:
        Model ready for evaluation
    """
    from .model_utils import replace_vit_layers_with_cim, wrap_fake_quant_modules

    model = build_empty_model(num_classes=num_classes)
    model = load_state_dict_to_model(model, base_state_dict)

    if quant_mode == 'CIM_Quant':
        model = replace_vit_layers_with_cim(
            model,
            parallel_read=parallel_read,
            variation_sigma=sigma,
            adc_bits=adc_bits,
            input_bits=input_bits,
            weight_bits=weight_bits,
            use_partial_sum_quant=use_partial_sum_quant,
            weight_encode_method=weight_encode_method,
            activation_encode_method=activation_encode_method,
            enable_activity_stats=enable_activity_stats,
            enable_sparsity_stats=enable_sparsity_stats
        )
    elif quant_mode == 'Fake_Quant':
        model = wrap_fake_quant_modules(
            model,
            sigma=sigma,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_encode_method=weight_encode_method,
            activation_encode_method=activation_encode_method,
            noise_enable=noise_enable,
            noise_mode=noise_mode,
            include_layers=include_layers,
            exclude_layers=exclude_layers
        )
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")

    model.to(device)
    model.eval()
    return model