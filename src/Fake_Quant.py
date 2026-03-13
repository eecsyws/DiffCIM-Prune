"""
Fake Quantization Layers

This module contains simplified CIM evaluation layers with noise injection:
- NoisyLinear: Linear layer with fake quantization and variation noise
- NoisyConv2d: Conv2d layer with fake quantization and variation noise
- Helper functions for quantization and noise injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_any(name: str, patterns):
    """Check if any pattern is in the name."""
    if not patterns:
        return False
    return any(p in name for p in patterns)


def _should_apply_noise(layer_name, noise_enable, noise_sigma, noise_mode,
                        include_layers, exclude_layers):
    """Determine if noise should be applied to a layer."""
    if (not noise_enable) or noise_sigma <= 0:
        return False

    if noise_mode == "include":
        return _match_any(layer_name, include_layers)

    if noise_mode == "exclude":
        return not _match_any(layer_name, exclude_layers)

    raise ValueError(
        f"Unsupported noise_mode: {noise_mode}. "
        "Please choose 'include' or 'exclude'."
    )


def act_dynamic_fake_quant(x, num_bits, granularity="per-token"):
    """Apply dynamic fake quantization to activations."""
    if num_bits >= 32:
        return x

    qmax = (2 ** (num_bits - 1)) - 1
    qmin = -(2 ** (num_bits - 1))

    if granularity == "per-token":
        reduce_dims = -1
    else:
        reduce_dims = [i for i in range(1, x.dim())]

    max_abs = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = max_abs / float(qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_q = torch.round(x / scale).clamp(qmin, qmax)
    return x_q * scale


def per_channel_fake_quant_weight(weight, ch_axis, num_bits, mode):
    """Apply per-channel fake quantization to weights."""
    if mode == "single":
        qmin, qmax = -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1
    elif mode == "differential":
        qmax = (2 ** (num_bits - 1)) - 1
        qmin = -qmax
    else:
        raise ValueError(
            f"Unsupported encoding mode: {mode}. "
            "Please choose 'single' or 'differential'."
        )

    permute_dims = [ch_axis] + [i for i in range(weight.dim()) if i != ch_axis]
    inv_permute = [permute_dims.index(i) for i in range(len(permute_dims))]
    w = weight.permute(permute_dims)

    max_abs = w.reshape(w.size(0), -1).abs().max(dim=1)[0]
    scale = max_abs / float(qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    view_shape = [w.size(0)] + [1] * (weight.dim() - 1)
    scale_b = scale.view(view_shape).to(weight.device)

    q = torch.round(w / scale_b).clamp(qmin, qmax)
    return q.permute(inv_permute).contiguous(), scale.to(weight.device)


def add_bitwise_noise_generalized(q_float, sigma, num_bits, mode):
    """
    Add bit-wise noise to quantized weights.

    For 'single' mode: applies noise to signed integer representation
    For 'differential' mode: applies noise to positive/negative split representation
    """
    if sigma <= 0:
        return q_float

    if mode == "single":
        q_int = torch.round(q_float).to(torch.int32)
        offset = 2 ** num_bits
        q_unsigned = torch.where(q_int < 0, q_int + offset, q_int).to(torch.int32)

        res = torch.zeros_like(q_float)
        for bit in range(num_bits):
            mask = 1 << bit
            bit_plane = ((q_unsigned & mask) >> bit).float()
            bit_weight = -float(1 << bit) if bit == num_bits - 1 else float(1 << bit)
            noise = (1.0 + (torch.randn_like(res) * sigma).clamp(-3 * sigma, 3 * sigma))
            res += bit_plane * noise * bit_weight
        return res

    if mode == "differential":
        q_int = torch.round(q_float).to(torch.int32)
        pos_part = torch.clamp(q_int, min=0)
        neg_part = torch.clamp(-q_int, min=0)

        p_noised = torch.zeros_like(q_float)
        n_noised = torch.zeros_like(q_float)

        for bit in range(num_bits - 1):
            mask = 1 << bit
            bit_weight = float(mask)

            for part, noised_sum in [(pos_part, p_noised), (neg_part, n_noised)]:
                bit_plane = ((part & mask) >> bit).float()
                noise = (1.0 + (torch.randn_like(q_float) * sigma).clamp(-3 * sigma, 3 * sigma))
                noised_sum += bit_plane * noise * bit_weight

        return p_noised - n_noised

    raise ValueError(
        f"Unsupported encoding mode: {mode}. "
        "Please choose 'single' or 'differential'."
    )


class NoisyLinear(nn.Linear):
    """
    Linear layer with fake quantization and variation noise injection.

    This is a simplified CIM evaluation layer that:
    - Applies dynamic fake quantization to inputs
    - Applies per-channel fake quantization to weights
    - Optionally injects bit-wise variation noise
    """
    def __init__(self, in_features, out_features, bias=True, layer_name="",
                 num_bits_weight=8, num_bits_act=8, encoding_mode="single",
                 noise_enable=True, noise_sigma=0.0, noise_mode="exclude",
                 include_layers=None, exclude_layers=None):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_name = layer_name
        self.num_bits_weight = num_bits_weight
        self.num_bits_act = num_bits_act
        self.encoding_mode = encoding_mode
        self.noise_enable = noise_enable
        self.noise_sigma = noise_sigma
        self.noise_mode = noise_mode
        self.include_layers = include_layers or []
        self.exclude_layers = exclude_layers or []

    def forward(self, x):
        x_q = act_dynamic_fake_quant(x, self.num_bits_act, granularity="per-token")

        if _should_apply_noise(
            self.layer_name,
            self.noise_enable,
            self.noise_sigma,
            self.noise_mode,
            self.include_layers,
            self.exclude_layers
        ):
            q, scale = per_channel_fake_quant_weight(
                self.weight, 0, self.num_bits_weight, self.encoding_mode
            )
            q_noised = add_bitwise_noise_generalized(
                q, self.noise_sigma, self.num_bits_weight, self.encoding_mode
            )
            w_used = q_noised * scale.view(-1, 1)
        else:
            w_used = self.weight

        return F.linear(x_q, w_used, self.bias)


class NoisyConv2d(nn.Conv2d):
    """
    Conv2d layer with fake quantization and variation noise injection.

    This is a simplified CIM evaluation layer that:
    - Applies dynamic fake quantization to inputs
    - Applies per-channel fake quantization to weights
    - Optionally injects bit-wise variation noise
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', layer_name="",
                 num_bits_weight=8, num_bits_act=8, encoding_mode="single",
                 noise_enable=True, noise_sigma=0.0, noise_mode="exclude",
                 include_layers=None, exclude_layers=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode
        )
        self.layer_name = layer_name
        self.num_bits_weight = num_bits_weight
        self.num_bits_act = num_bits_act
        self.encoding_mode = encoding_mode
        self.noise_enable = noise_enable
        self.noise_sigma = noise_sigma
        self.noise_mode = noise_mode
        self.include_layers = include_layers or []
        self.exclude_layers = exclude_layers or []

    def forward(self, x):
        x_q = act_dynamic_fake_quant(x, self.num_bits_act, granularity="per-sample")

        if _should_apply_noise(
            self.layer_name,
            self.noise_enable,
            self.noise_sigma,
            self.noise_mode,
            self.include_layers,
            self.exclude_layers
        ):
            q, scale = per_channel_fake_quant_weight(
                self.weight, 0, self.num_bits_weight, self.encoding_mode
            )
            q_noised = add_bitwise_noise_generalized(
                q, self.noise_sigma, self.num_bits_weight, self.encoding_mode
            )
            w_used = q_noised * scale.view(1, -1, 1, 1)
        else:
            w_used = self.weight

        return F.conv2d(
            x_q, w_used, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
