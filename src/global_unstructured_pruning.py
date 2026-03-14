"""
Global Unstructured Pruning Module

This module contains all pruning-related functions:
- Prunable layer detection
- Global magnitude-based pruning
- State dict preparation
"""

import torch
import torch.nn as nn


def is_prunable_linear_layer(layer_name, layer_module, exclude_keywords):
    """
    Return True only for Linear layers that are allowed to be pruned.

    Args:
        layer_name: Name of the layer
        layer_module: The layer module
        exclude_keywords: List of keywords to exclude

    Returns:
        True if the layer is prunable, False otherwise
    """
    if not isinstance(layer_module, nn.Linear):
        return False

    if any(keyword in layer_name for keyword in exclude_keywords):
        return False

    return True


def collect_prunable_linear_layers(model, exclude_keywords):
    """
    Collect all eligible Linear layers for global pruning.

    Args:
        model: The model to collect prunable layers from
        exclude_keywords: List of keywords to exclude from pruning

    Returns:
        List of (name, module) tuples for prunable layers
    """
    prunable = []
    for name, module in model.named_modules():
        if is_prunable_linear_layer(name, module, exclude_keywords):
            prunable.append((name, module))
    return prunable


def apply_global_unstructured_pruning(model, pruning_rate, exclude_keywords):
    """
    Apply global unstructured magnitude pruning to eligible Linear weights.

    Rule:
    - gather all eligible Linear.weight values globally
    - sort by absolute magnitude
    - set the smallest global fraction to zero

    Args:
        model: The model to prune
        pruning_rate: Fraction of weights to prune (0.0 - 1.0)
        exclude_keywords: List of keywords to exclude from pruning

    Returns:
        A stats dict for logging
    """
    prunable_layers = collect_prunable_linear_layers(model, exclude_keywords)

    if len(prunable_layers) == 0:
        print("[Pruning] No eligible Linear layers found. Skip pruning.")
        return {
            'pruned_layers': 0,
            'total_weights': 0,
            'pruned_weights': 0,
            'sparsity': 0.0
        }

    flat_abs_list = []
    layer_meta = []

    for name, module in prunable_layers:
        weight_abs = module.weight.detach().abs().reshape(-1).cpu()
        flat_abs_list.append(weight_abs)
        layer_meta.append((name, module, module.weight.shape, weight_abs.numel()))

    global_abs = torch.cat(flat_abs_list, dim=0)
    total_weights = global_abs.numel()
    pruned_weights = int(total_weights * pruning_rate)

    if pruned_weights <= 0:
        print("[Pruning] PRUNING_RATE is 0. No weights were pruned.")
        return {
            'pruned_layers': len(prunable_layers),
            'total_weights': total_weights,
            'pruned_weights': 0,
            'sparsity': 0.0
        }

    global_keep_mask = torch.ones(total_weights, dtype=torch.bool)
    prune_indices = torch.topk(global_abs, k=pruned_weights, largest=False).indices
    global_keep_mask[prune_indices] = False

    cursor = 0
    actual_zero_count = 0

    with torch.no_grad():
        for name, module, weight_shape, numel in layer_meta:
            local_keep_mask = global_keep_mask[cursor:cursor + numel].view(weight_shape)
            local_keep_mask = local_keep_mask.to(
                device=module.weight.device,
                dtype=module.weight.dtype
            )
            module.weight.data.mul_(local_keep_mask)
            actual_zero_count += int((module.weight.data == 0).sum().item())
            cursor += numel

    sparsity = actual_zero_count / float(total_weights)

    print(
        f"[Pruning] Applied global unstructured pruning on {len(prunable_layers)} Linear layers. "
        f"Target prune ratio={pruning_rate:.4f}, actual sparsity={sparsity:.4f} "
        f"({actual_zero_count}/{total_weights})."
    )

    return {
        'pruned_layers': len(prunable_layers),
        'total_weights': total_weights,
        'pruned_weights': actual_zero_count,
        'sparsity': sparsity
    }


def clone_state_dict_to_cpu(model):
    """
    Clone model weights to a pure CPU state_dict.

    This is the in-memory temporary checkpoint used for later experiments.

    Args:
        model: The model to clone state dict from

    Returns:
        CPU state dict
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def prepare_base_state_dict_for_run(model_loader_fn, pruning_enable, pruning_rate, exclude_keywords):
    """
    Prepare the base state_dict used by all sigma experiments.

    Order:
    1. load original model
    2. optionally prune
    3. keep the pruned result in memory only
    4. do NOT save any temporary checkpoint to local disk

    Args:
        model_loader_fn: Function to load the model from checkpoint
        pruning_enable: Whether to enable pruning
        pruning_rate: Pruning rate (0.0 - 1.0)
        exclude_keywords: Keywords to exclude from pruning

    Returns:
        - base_state_dict: CPU state_dict kept in memory
        - pruning_stats: pruning summary or None
    """
    if (not pruning_enable) or pruning_rate <= 0:
        print("[Pruning] Disabled. Use original checkpoint directly in memory.")
        model = model_loader_fn()
        base_state_dict = clone_state_dict_to_cpu(model)
        del model
        import gc
        gc.collect()
        return base_state_dict, None

    base_model = model_loader_fn()
    pruning_stats = apply_global_unstructured_pruning(base_model, pruning_rate, exclude_keywords)
    base_state_dict = clone_state_dict_to_cpu(base_model)

    print("[Pruning] Temporary pruned weights kept in memory only. No local file will be saved.")

    del base_model
    import gc
    gc.collect()

    return base_state_dict, pruning_stats