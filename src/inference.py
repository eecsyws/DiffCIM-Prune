import gc
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Import from src package
from .model_loader import (
    build_empty_model,
    load_checkpoint_to_model,
    load_state_dict_to_model,
    load_model_from_checkpoint,
    build_model_for_current_mode,
)

from .global_unstructured_pruning import prepare_base_state_dict_for_run

# Import config
from config import config as cfg

# ==============================================================================
# Section 0: Utility Functions
# ==============================================================================

def setup_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Make CuDNN deterministic for stable repeated experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_activity_stats(model):
    """
    Collect activity statistics from CIM layers.

    Returns:
        dict with keys: total_1x1_count, total_1_count, total_weight_1_count
    """
    total_1x1 = 0
    total_1 = 0
    total_w1 = 0

    for module in model.modules():
        if hasattr(module, 'enable_activity_stats') and module.enable_activity_stats:
            total_1x1 += module.total_1x1_count
            total_1 += module.total_1_count
            total_w1 += module.total_weight_1_count

    return {
        'total_1x1_count': total_1x1,
        'total_1_count': total_1,
        'total_weight_1_count': total_w1
    }


def collect_sparsity_stats(model):
    """
    Collect per-bit-plane density statistics from CIM layers.
    Density = probability of bit being 1.

    Returns:
        dict with keys:
        - weight_density: list of (bit_idx, density) for each weight bit plane
        - activation_density: list of (bit_idx, density) for each activation bit plane
    """
    all_weight_density = []
    all_activation_density = []

    for module in model.modules():
        if hasattr(module, 'enable_sparsity_stats') and module.enable_sparsity_stats:
            if hasattr(module, 'weight_density') and module.weight_density is not None:
                layer_name = getattr(module, 'layer_name', None)
                if layer_name is None:
                    # Try to get name from parent's children
                    layer_name = module.name if hasattr(module, 'name') else 'unknown'
                all_weight_density.append({
                    'layer': layer_name,
                    'data': module.weight_density
                })
            if hasattr(module, 'activation_density') and module.activation_density is not None:
                layer_name = getattr(module, 'layer_name', None)
                if layer_name is None:
                    layer_name = module.name if hasattr(module, 'name') else 'unknown'
                all_activation_density.append({
                    'layer': layer_name,
                    'data': module.activation_density
                })

    return {
        'weight_density': all_weight_density,
        'activation_density': all_activation_density
    }


def maybe_autocast(device_str):
    """
    Return an autocast context only when CUDA is available.
    On CPU, return a no-op context.
    """
    if device_str == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def validate_config():
    """Check whether the main config options are valid."""
    if cfg.QUANT_MODE not in ['CIM_Quant', 'Fake_Quant']:
        raise ValueError(
            f"Unsupported quant_mode: {cfg.QUANT_MODE}. "
            "Please choose 'CIM_Quant' or 'Fake_Quant'."
        )

    if cfg.WEIGHT_ENCODE_METHOD not in ['twos_complement', 'differential']:
        raise ValueError(
            f"Unsupported WEIGHT_ENCODE_METHOD: {cfg.WEIGHT_ENCODE_METHOD}. "
            "Please choose 'twos_complement' or 'differential'."
        )

    if cfg.ACTIVATION_ENCODE_METHOD not in ['twos_complement', 'differential']:
        raise ValueError(
            f"Unsupported ACTIVATION_ENCODE_METHOD: {cfg.ACTIVATION_ENCODE_METHOD}. "
            "Please choose 'twos_complement' or 'differential'."
        )

    if cfg.NOISE_MODE not in ['include', 'exclude']:
        raise ValueError(
            f"Unsupported NOISE_MODE: {cfg.NOISE_MODE}. "
            "Please choose 'include' or 'exclude'."
        )

    if not (0.0 <= cfg.PRUNING_RATE <= 1.0):
        raise ValueError(
            f"Unsupported PRUNING_RATE: {cfg.PRUNING_RATE}. "
            "Please choose a value in [0.0, 1.0]."
        )


# ==============================================================================
# Section 1: Global Config (using config.py)
# ==============================================================================

# Aliases for backward compatibility in this module
MODEL_PATH = cfg.MODEL_PATH
DATASET_DIR = cfg.DATASET_DIR
BATCH_SIZE = cfg.BATCH_SIZE
NUM_WORKERS = cfg.NUM_WORKERS
DEVICE = cfg.DEVICE
NUM_CLASSES = cfg.NUM_CLASSES
SEED = cfg.SEED
MAX_TEST_SAMPLES = cfg.MAX_TEST_SAMPLES

print(f"[Info] Random seed set to {SEED}")

quant_mode = cfg.QUANT_MODE
encode_method = cfg.WEIGHT_ENCODE_METHOD
activation_encode_method = cfg.ACTIVATION_ENCODE_METHOD
WEIGHT_BITS = cfg.WEIGHT_BITS
INPUT_BITS = cfg.INPUT_BITS
ADC_BITS = cfg.ADC_BITS
parallel_read = cfg.PARALLEL_READ
USE_PARTIAL_SUM_QUANT = cfg.USE_PARTIAL_SUM_QUANT
NOISE_ENABLE = cfg.NOISE_ENABLE
NOISE_MODE = cfg.NOISE_MODE
INCLUDE_LAYERS = cfg.INCLUDE_LAYERS
EXCLUDE_LAYERS = cfg.EXCLUDE_LAYERS
VARIATION_SIGMA_LIST = cfg.VARIATION_SIGMA_LIST
PRUNING_ENABLE = cfg.PRUNING_ENABLE
PRUNING_RATE = cfg.PRUNING_RATE
PRUNING_EXCLUDE_KEYWORDS = cfg.PRUNING_EXCLUDE_KEYWORDS
ENABLE_ACTIVITY_STATS = cfg.ENABLE_ACTIVITY_STATS
ENABLE_SPARSITY_STATS = cfg.ENABLE_SPARSITY_STATS
CIFAR100_MEAN = cfg.CIFAR100_MEAN
CIFAR100_STD = cfg.CIFAR100_STD


# ==============================================================================
# Section 2: Main Evaluation Loop
# ==============================================================================

def run_inference():
    """Main inference function."""
    validate_config()
    setup_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Quant mode: {quant_mode}")
    print(f"Weight encode method: {encode_method}")
    print(f"Activation encode method: {activation_encode_method}")
    print("Preparing Dataset...")

    test_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=test_tf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda')
    )

    # Define model loader function for pruning
    def model_loader_fn():
        return load_model_from_checkpoint(MODEL_PATH, num_classes=NUM_CLASSES)

    # Prepare the base state_dict once.
    base_state_dict, pruning_stats = prepare_base_state_dict_for_run(
        model_loader_fn,
        PRUNING_ENABLE,
        PRUNING_RATE,
        PRUNING_EXCLUDE_KEYWORDS
    )

    results = []

    print(f"\n{'=' * 100}")
    print("Starting Quantized Inference Sweep")
    print(
        f"Config: QuantMode={quant_mode} | W_Encode={encode_method} | A_Encode={activation_encode_method} | "
        f"W{WEIGHT_BITS} A{INPUT_BITS} | Max Samples={MAX_TEST_SAMPLES}"
    )

    if quant_mode == 'CIM_Quant':
        print(
            f"CIM Config: ADC={ADC_BITS} | ParallelRead={parallel_read} | "
            f"PSum Quant={USE_PARTIAL_SUM_QUANT}"
        )
    else:
        print(
            f"FakeQuant Config: NoiseEnable={NOISE_ENABLE} | NoiseMode={NOISE_MODE} | "
            f"ExcludeLayers={EXCLUDE_LAYERS}"
        )

    print(
        f"Pruning Config: Enable={PRUNING_ENABLE} | Rate={PRUNING_RATE} | "
        f"Exclude={PRUNING_EXCLUDE_KEYWORDS}"
    )
    if pruning_stats is not None:
        print(
            f"Pruning Result: Layers={pruning_stats['pruned_layers']} | "
            f"Weights={pruning_stats['pruned_weights']}/{pruning_stats['total_weights']} | "
            f"Sparsity={pruning_stats['sparsity']:.4f}"
        )

    print("Base weights used for sweep: in-memory temporary state_dict")
    print(f"Sigma Sweep List: {VARIATION_SIGMA_LIST}")
    print(f"{'=' * 100}")

    exp_idx = 0

    for sigma in VARIATION_SIGMA_LIST:
        exp_idx += 1
        config_str = f"{quant_mode}_W{encode_method}_A{activation_encode_method}_P{PRUNING_RATE}_S{sigma}"
        print(f"\n[Exp {exp_idx}] Config: {config_str}")

        model = build_model_for_current_mode(
            sigma=sigma,
            base_state_dict=base_state_dict,
            quant_mode=quant_mode,
            weight_encode_method=encode_method,
            activation_encode_method=activation_encode_method,
            weight_bits=WEIGHT_BITS,
            input_bits=INPUT_BITS,
            adc_bits=ADC_BITS,
            parallel_read=parallel_read,
            use_partial_sum_quant=USE_PARTIAL_SUM_QUANT,
            noise_enable=NOISE_ENABLE,
            noise_mode=NOISE_MODE,
            include_layers=INCLUDE_LAYERS,
            exclude_layers=EXCLUDE_LAYERS,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            enable_activity_stats=ENABLE_ACTIVITY_STATS,
            enable_sparsity_stats=ENABLE_SPARSITY_STATS
        )

        correct = 0
        total = 0
        early_stopped_bad_acc = False

        pbar = tqdm(test_loader, desc=config_str, leave=False)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with maybe_autocast(DEVICE):
                    outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch_idx == 0:
                    current_acc = 100 * correct / total
                    if current_acc < 2.0:
                        print(f"    [Warning] Early Stop! Acc {current_acc:.2f}% < 2%.")
                        early_stopped_bad_acc = True
                        break

                if total >= MAX_TEST_SAMPLES:
                    break

        final_acc = 100 * correct / total

        if early_stopped_bad_acc:
            status_note = 'Bad_Acc(<2%)'
        elif total >= MAX_TEST_SAMPLES:
            status_note = f'Limit({total})'
        else:
            status_note = 'Full_Set'

        # Collect activity statistics if enabled
        activity_stats = None
        if ENABLE_ACTIVITY_STATS:
            activity_stats = collect_activity_stats(model)
            print(f"    Activity: 1x1={activity_stats['total_1x1_count']:.0f}, "
                  f"Act_1={activity_stats['total_1_count']:.0f}, "
                  f"Weight_1={activity_stats['total_weight_1_count']:.0f}")

        # Collect density statistics if enabled (density = probability of bit=1)
        density_stats = None
        avg_weight_density = 'N/A'
        avg_activation_density = 'N/A'
        if ENABLE_SPARSITY_STATS:
            density_stats = collect_sparsity_stats(model)

            # Compute average density across all layers
            if density_stats['weight_density']:
                all_w_density = []
                for ws in density_stats['weight_density']:
                    all_w_density.extend([d for _, d in ws['data']])
                avg_weight_density = sum(all_w_density) / len(all_w_density) if all_w_density else 'N/A'
                # Print weight density per bit plane for ALL layers
                print(f"    Weight Density (per bit, avg={avg_weight_density:.3f}):")
                for ws in density_stats['weight_density']:
                    data = ws['data']
                    # If even number of bits (differential encoding), split into + and -
                    if len(data) % 2 == 0:
                        half = len(data) // 2
                        pos_bits = [f"bit{i}+={d:.3f}" for i, (_, d) in enumerate(data[:half])]
                        neg_bits = [f"bit{i}-={d:.3f}" for i, (_, d) in enumerate(data[half:])]
                        print(f"      {ws['layer']}: " + ", ".join(pos_bits + neg_bits))
                    else:
                        print(f"      {ws['layer']}: " + ", ".join([f"bit{b}={d:.3f}" for b, d in data]))

            if density_stats['activation_density']:
                all_a_density = []
                for as_ in density_stats['activation_density']:
                    all_a_density.extend([d for _, d in as_['data']])
                avg_activation_density = sum(all_a_density) / len(all_a_density) if all_a_density else 'N/A'
                # Print activation density per bit plane for ALL layers
                print(f"    Activation Density (per bit, avg={avg_activation_density:.3f}):")
                for as_ in density_stats['activation_density']:
                    data = as_['data']
                    # If even number of bits (differential encoding), split into + and -
                    if len(data) % 2 == 0:
                        half = len(data) // 2
                        pos_bits = [f"bit{i}+={d:.3f}" for i, (_, d) in enumerate(data[:half])]
                        neg_bits = [f"bit{i}-={d:.3f}" for i, (_, d) in enumerate(data[half:])]
                        print(f"      {as_['layer']}: " + ", ".join(pos_bits + neg_bits))
                    else:
                        print(f"      {as_['layer']}: " + ", ".join([f"bit{b}={d:.3f}" for b, d in data]))

        results.append({
            'QuantMode': quant_mode,
            'WeightEncodeMethod': encode_method,
            'ActivationEncodeMethod': activation_encode_method,
            'PruningRate': PRUNING_RATE if PRUNING_ENABLE else 0.0,
            'ParallelRead': parallel_read if quant_mode == 'CIM_Quant' else 'N/A',
            'ADC': ADC_BITS if quant_mode == 'CIM_Quant' else 'N/A',
            'Sigma': sigma,
            'Accuracy': final_acc,
            'Total_1x1': activity_stats['total_1x1_count'] if activity_stats else 'N/A',
            'Total_Act_1': activity_stats['total_1_count'] if activity_stats else 'N/A',
            'Total_Weight_1': activity_stats['total_weight_1_count'] if activity_stats else 'N/A',
            'AvgWeightDensity': avg_weight_density,
            'AvgActivationDensity': avg_activation_density,
            'Note': status_note
        })

        print(f"-> Result: {final_acc:.2f}% (Samples: {total})")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 140}")
    print(f"Final Summary: Quantized Inference Accuracy (Max {MAX_TEST_SAMPLES} Samples)")
    print(f"Quant Mode: {quant_mode}")
    print(f"Weight Encode Method: {encode_method}")
    print(f"Activation Encode Method: {activation_encode_method}")
    print(f"Pruning Enabled: {PRUNING_ENABLE}")
    print(f"Pruning Rate: {PRUNING_RATE if PRUNING_ENABLE else 0.0}")
    print(f"{'=' * 140}")
    print(
        f"{'QuantMode':<12} | {'W_Encode':<14} | {'A_Encode':<14} | {'Pruning':<8} | {'ParallelRead':<12} | "
        f"{'ADC':<5} | {'Sigma':<8} | {'Accuracy':<10} | {'Status'}"
    )
    print("-" * 150)

    for res in results:
        print(
            f"{res['QuantMode']:<12} | {res['WeightEncodeMethod']:<14} | {res['ActivationEncodeMethod']:<14} | "
            f"{res['PruningRate']:<8} | {str(res['ParallelRead']):<12} | "
            f"{str(res['ADC']):<5} | {res['Sigma']:<8} | "
            f"{res['Accuracy']:.2f}%     | {res['Note']}"
        )

    return results