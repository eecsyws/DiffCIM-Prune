"""
CIM Quantization Layers

This module contains CIM (Compute-In-Memory) related layers:
- CIM_Linear: Standard CIM linear layer
- CIM_SM_Linear: CIM linear layer with differential (signed-magnitude) encoding
"""

import torch
import torch.nn as nn
from contextlib import nullcontext


def _maybe_autocast(x):
    if torch.is_tensor(x) and x.is_cuda:
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


class CIM_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 input_bits=8, weight_bits=8, adc_bits=10,
                 rows_parallel=64, variation_sigma=0.0,
                 use_partial_sum_quant=False,
                 activation_encode_method="twos_complement"):
        super(CIM_Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.rows_parallel = rows_parallel
        self.variation_sigma = variation_sigma
        self.use_partial_sum_quant = use_partial_sum_quant
        self.activation_encode_method = activation_encode_method

        # For differential activation encoding, effective input bits is input_bits - 1
        if self.activation_encode_method == "differential":
            self.input_storage_bits = input_bits - 1
        else:
            self.input_storage_bits = input_bits

        if self.use_partial_sum_quant:
            self.alpha = self.rows_parallel / (2 ** self.adc_bits)
        else:
            self.alpha = 1.0

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def int2bit(self, tensor, bits, is_signed=False):
        tensor_int = tensor.to(torch.int32)
        if is_signed:
            mask_val = 2 ** bits
            tensor_int = torch.where(tensor_int < 0, tensor_int + mask_val, tensor_int)

        mask = 2 ** torch.arange(bits, device=tensor.device).reshape(-1, *([1] * tensor.ndim))
        return (tensor_int.unsqueeze(0) & mask).bool().float()

    def adc_quantize(self, analog_sum):
        max_val = 2 ** self.adc_bits - 1
        if self.use_partial_sum_quant:
            scaled_sum = analog_sum.float() / self.alpha
            adc_code = torch.clamp(scaled_sum.round(), 0, max_val)
            return adc_code * self.alpha
        return torch.clamp(analog_sum.float(), 0, max_val).round()

    def apply_variation(self, w_bits):
        if self.variation_sigma <= 0:
            return w_bits
        noise = torch.randn_like(w_bits) * self.variation_sigma
        return torch.clamp(w_bits * (1.0 + noise), min=0.0)

    def fake_quantize_input_per_token(self, x):
        max_val = x.abs().amax(dim=1, keepdim=True)
        q_max = 2 ** (self.input_bits - 1) - 1
        scale = max_val / q_max
        min_int = -(2 ** (self.input_bits - 1))
        max_int = q_max
        x_q = (x / (scale + 1e-8)).round().clamp(min_int, max_int)
        return x_q, scale

    def fake_quantize_input_per_token_differential(self, x):
        """
        Quantize input for differential activation encoding.

        Returns:
            x_pos: positive part (max(x_q, 0))
            x_neg: negative part (max(-x_q, 0))
            scale: scale factor
        """
        max_val = x.abs().amax(dim=1, keepdim=True)
        # For differential, effective bits is input_bits - 1, range [0, 2^(n-1)-1]
        q_max = 2 ** (self.input_storage_bits) - 1
        scale = max_val / q_max
        # Quantize to signed integer first
        min_int = -(2 ** (self.input_bits - 1))
        max_int = q_max
        x_q = (x / (scale + 1e-8)).round().clamp(min_int, max_int)
        # Split into positive and negative parts (both non-negative)
        x_pos = torch.relu(x_q)
        x_neg = torch.relu(-x_q)
        return x_pos, x_neg, scale

    def fake_quantize_weight_per_channel(self, w):
        max_val = w.abs().amax(dim=1, keepdim=True)
        q_max = 2 ** (self.weight_bits - 1) - 1
        scale = max_val / q_max
        min_int = -(2 ** (self.weight_bits - 1))
        max_int = q_max
        w_q = (w / (scale + 1e-8)).round().clamp(min_int, max_int)
        return w_q, scale

    def forward(self, input):
        origin_shape = input.shape
        x_flat = input.reshape(-1, self.in_features)

        w_q, scale_w = self.fake_quantize_weight_per_channel(self.weight)

        total_tokens, _ = x_flat.shape
        out_channels, _ = w_q.shape

        # Handle activation encoding based on method
        if self.activation_encode_method == "differential":
            x_pos, x_neg, scale_x = self.fake_quantize_input_per_token_differential(x_flat)
            # Convert to bit planes (unsigned for both positive and negative)
            X_pos_planes = self.int2bit(x_pos, self.input_storage_bits, is_signed=False)
            X_neg_planes = self.int2bit(x_neg, self.input_storage_bits, is_signed=False)
        else:
            x_q, scale_x = self.fake_quantize_input_per_token(x_flat)
            X_planes = self.int2bit(x_q, self.input_bits, is_signed=True)
            X_pos_planes = None
            X_neg_planes = None

        W_planes = self.int2bit(w_q, self.weight_bits, is_signed=True)

        if self.variation_sigma > 0:
            W_planes = self.apply_variation(W_planes)

        output_int = torch.zeros((total_tokens, out_channels), device=input.device)

        for w_bit in range(self.weight_bits):
            w_plane = W_planes[w_bit]
            w_bit_partial = torch.zeros_like(output_int)

            for row_start in range(0, self.in_features, self.rows_parallel):
                row_end = min(row_start + self.rows_parallel, self.in_features)

                w_chunk = w_plane[:, row_start:row_end]
                row_chunk_sum = torch.zeros_like(output_int)

                if self.activation_encode_method == "differential":
                    # Differential activation: compute (X_pos - X_neg) * W
                    x_pos_chunk_all = X_pos_planes[:, :, row_start:row_end]
                    x_neg_chunk_all = X_neg_planes[:, :, row_start:row_end]

                    for x_bit in range(self.input_storage_bits):
                        # Positive part contribution
                        x_pos_chunk_bit = x_pos_chunk_all[x_bit]
                        with _maybe_autocast(x_pos_chunk_bit):
                            analog_val_pos = torch.matmul(x_pos_chunk_bit, w_chunk.t())
                        digital_val_pos = self.adc_quantize(analog_val_pos)

                        # Negative part contribution (subtracted)
                        x_neg_chunk_bit = x_neg_chunk_all[x_bit]
                        with _maybe_autocast(x_neg_chunk_bit):
                            analog_val_neg = torch.matmul(x_neg_chunk_bit, w_chunk.t())
                        digital_val_neg = self.adc_quantize(analog_val_neg)

                        # Differential: positive - negative
                        row_chunk_sum += (digital_val_pos - digital_val_neg) * (2 ** x_bit)
                else:
                    # Standard two's complement activation
                    x_chunk_all = X_planes[:, :, row_start:row_end]

                    for x_bit in range(self.input_bits):
                        x_chunk_bit = x_chunk_all[x_bit]

                        with _maybe_autocast(x_chunk_bit):
                            analog_val = torch.matmul(x_chunk_bit, w_chunk.t())

                        digital_val = self.adc_quantize(analog_val)

                        if x_bit == (self.input_bits - 1):
                            row_chunk_sum -= digital_val * (2 ** x_bit)
                        else:
                            row_chunk_sum += digital_val * (2 ** x_bit)

                w_bit_partial += row_chunk_sum

            if w_bit == (self.weight_bits - 1):
                output_int -= w_bit_partial * (2 ** w_bit)
            else:
                output_int += w_bit_partial * (2 ** w_bit)

        output_fp32 = output_int * scale_x * scale_w.t()
        output_final = output_fp32.reshape(*origin_shape[:-1], self.out_features)

        if self.bias is not None:
            output_final += self.bias

        return output_final


class CIM_SM_Linear(nn.Module):
    """
    CIM Linear layer with differential (signed-magnitude) encoding.

    Splits weights into positive (w_pos) and negative (w_neg) components,
    computes them separately, and subtracts to get the final result.
    This reduces variation impact on high-weight bits.
    """
    def __init__(self, in_features, out_features, bias=True,
                 input_bits=8, weight_bits=8, adc_bits=10,
                 rows_parallel=64, variation_sigma=0.0,
                 use_partial_sum_quant=False,
                 activation_encode_method="twos_complement"):
        super(CIM_SM_Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.weight_storage_bits = weight_bits - 1
        self.adc_bits = adc_bits
        self.rows_parallel = rows_parallel
        self.variation_sigma = variation_sigma
        self.use_partial_sum_quant = use_partial_sum_quant
        self.activation_encode_method = activation_encode_method

        # For differential activation encoding, effective input bits is input_bits - 1
        if self.activation_encode_method == "differential":
            self.input_storage_bits = input_bits - 1
        else:
            self.input_storage_bits = input_bits

        if self.use_partial_sum_quant:
            self.alpha = self.rows_parallel / (2 ** self.adc_bits)
        else:
            self.alpha = 1.0

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def int2bit(self, tensor, bits, is_signed=False):
        tensor_int = tensor.to(torch.int32)
        if is_signed:
            mask_val = 2 ** bits
            tensor_int = torch.where(tensor_int < 0, tensor_int + mask_val, tensor_int)

        mask = 2 ** torch.arange(bits, device=tensor.device).reshape(-1, *([1] * tensor.ndim))
        return (tensor_int.unsqueeze(0) & mask).bool().float()

    def adc_quantize(self, analog_sum):
        max_val = 2 ** self.adc_bits - 1
        if self.use_partial_sum_quant:
            scaled_sum = analog_sum.float() / self.alpha
            adc_code = torch.clamp(scaled_sum.round(), 0, max_val)
            return adc_code * self.alpha
        return torch.clamp(analog_sum.float(), 0, max_val).round()

    def apply_variation(self, w_bits):
        if self.variation_sigma <= 0:
            return w_bits
        noise = torch.randn_like(w_bits) * self.variation_sigma
        return torch.clamp(w_bits * (1.0 + noise), min=0.0)

    def fake_quantize_input_per_token(self, x):
        max_val = x.abs().amax(dim=1, keepdim=True)
        q_max = 2 ** (self.input_bits - 1) - 1
        scale = max_val / q_max
        min_int = -(2 ** (self.input_bits - 1))
        max_int = q_max
        x_q = (x / (scale + 1e-8)).round().clamp(min_int, max_int)
        return x_q, scale

    def fake_quantize_input_per_token_differential(self, x):
        """
        Quantize input for differential activation encoding.

        Returns:
            x_pos: positive part (max(x_q, 0))
            x_neg: negative part (max(-x_q, 0))
            scale: scale factor
        """
        max_val = x.abs().amax(dim=1, keepdim=True)
        # For differential, effective bits is input_bits - 1, range [0, 2^(n-1)-1]
        q_max = 2 ** (self.input_storage_bits) - 1
        scale = max_val / q_max
        # Quantize to signed integer first
        min_int = -(2 ** (self.input_bits - 1))
        max_int = q_max
        x_q = (x / (scale + 1e-8)).round().clamp(min_int, max_int)
        # Split into positive and negative parts (both non-negative)
        x_pos = torch.relu(x_q)
        x_neg = torch.relu(-x_q)
        return x_pos, x_neg, scale

    def fake_quantize_weight_per_channel_split(self, w):
        """Split weight into positive and negative parts for differential encoding."""
        max_val = w.abs().amax(dim=1, keepdim=True)
        q_max = 2 ** (self.weight_bits - 1) - 1
        scale = max_val / q_max
        min_int = -q_max
        max_int = q_max

        w_q = (w / (scale + 1e-8)).round().clamp(min_int, max_int)
        w_pos = torch.relu(w_q)
        w_neg = torch.relu(-w_q)
        return w_pos, w_neg, scale

    def forward(self, input):
        origin_shape = input.shape
        x_flat = input.reshape(-1, self.in_features)

        w_pos, w_neg, scale_w = self.fake_quantize_weight_per_channel_split(self.weight)

        # Stack positive and negative weights for parallel computation
        w_stack = torch.cat([w_pos, w_neg], dim=0)
        total_tokens, _ = x_flat.shape
        out_channels_doubled, _ = w_stack.shape

        # Handle activation encoding based on method
        if self.activation_encode_method == "differential":
            x_pos, x_neg, scale_x = self.fake_quantize_input_per_token_differential(x_flat)
            # Convert to bit planes (unsigned for both positive and negative)
            X_pos_planes = self.int2bit(x_pos, self.input_storage_bits, is_signed=False)
            X_neg_planes = self.int2bit(x_neg, self.input_storage_bits, is_signed=False)
        else:
            x_q, scale_x = self.fake_quantize_input_per_token(x_flat)
            X_planes = self.int2bit(x_q, self.input_bits, is_signed=True)
            X_pos_planes = None
            X_neg_planes = None

        W_planes = self.int2bit(w_stack, self.weight_storage_bits, is_signed=False)

        if self.variation_sigma > 0:
            W_planes = self.apply_variation(W_planes)

        output_stack = torch.zeros((total_tokens, out_channels_doubled), device=input.device)

        for w_bit in range(self.weight_storage_bits):
            w_plane = W_planes[w_bit]
            w_bit_partial = torch.zeros_like(output_stack)

            for row_start in range(0, self.in_features, self.rows_parallel):
                row_end = min(row_start + self.rows_parallel, self.in_features)

                w_chunk = w_plane[:, row_start:row_end]
                row_chunk_sum = torch.zeros_like(output_stack)

                if self.activation_encode_method == "differential":
                    # Differential activation: compute (X_pos - X_neg) * W
                    x_pos_chunk_all = X_pos_planes[:, :, row_start:row_end]
                    x_neg_chunk_all = X_neg_planes[:, :, row_start:row_end]

                    for x_bit in range(self.input_storage_bits):
                        # Positive part contribution
                        x_pos_chunk_bit = x_pos_chunk_all[x_bit]
                        with _maybe_autocast(x_pos_chunk_bit):
                            analog_val_pos = torch.matmul(x_pos_chunk_bit, w_chunk.t())
                        digital_val_pos = self.adc_quantize(analog_val_pos)

                        # Negative part contribution (subtracted)
                        x_neg_chunk_bit = x_neg_chunk_all[x_bit]
                        with _maybe_autocast(x_neg_chunk_bit):
                            analog_val_neg = torch.matmul(x_neg_chunk_bit, w_chunk.t())
                        digital_val_neg = self.adc_quantize(analog_val_neg)

                        # Differential: positive - negative
                        row_chunk_sum += (digital_val_pos - digital_val_neg) * (2 ** x_bit)
                else:
                    # Standard two's complement activation
                    x_chunk_all = X_planes[:, :, row_start:row_end]

                    for x_bit in range(self.input_bits):
                        x_chunk_bit = x_chunk_all[x_bit]

                        with _maybe_autocast(x_chunk_bit):
                            analog_val = torch.matmul(x_chunk_bit, w_chunk.t())

                        digital_val = self.adc_quantize(analog_val)

                        if x_bit == (self.input_bits - 1):
                            row_chunk_sum -= digital_val * (2 ** x_bit)
                        else:
                            row_chunk_sum += digital_val * (2 ** x_bit)

                w_bit_partial += row_chunk_sum

            output_stack += w_bit_partial * (2 ** w_bit)

        # Split and subtract to get final output
        out_pos, out_neg = torch.chunk(output_stack, 2, dim=1)
        output_int = out_pos - out_neg

        output_fp32 = output_int * scale_x * scale_w.t()
        output_final = output_fp32.reshape(*origin_shape[:-1], self.out_features)

        if self.bias is not None:
            output_final += self.bias

        return output_final