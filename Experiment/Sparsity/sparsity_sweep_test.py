"""
Sparsity Sweep Experiment Script for DiffCIM-Prune

This script runs a sweep over:
- Weight encoding: twos_complement / differential
- Activation encoding: twos_complement / differential
- Pruning rate: 0, 0.1, 0.2, 0.3, 0.4
- Sigma: 0 (fixed for sparsity analysis)

Total: 4 x 5 = 20 experiments

Outputs per-bit-plane sparsity statistics (weight & activation density).
"""

import os
import sys
import gc
import json
import time
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import config
from config import config as cfg

# Import inference components
from src.model_loader import (
    load_model_from_checkpoint,
    build_model_for_current_mode,
)
from src.global_unstructured_pruning import prepare_base_state_dict_for_run
from src.inference import setup_seed, maybe_autocast, CIFAR100_MEAN, CIFAR100_STD, collect_sparsity_stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# ==============================================================================
# Experiment Parameters
# ==============================================================================

# Encoding combinations: (weight_encode, activation_encode)
ENCODING_COMBOS = [
    ('twos_complement', 'twos_complement'),
    ('twos_complement', 'differential'),
    ('differential', 'twos_complement'),
    ('differential', 'differential'),
]

# Pruning rates
PRUNING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# Sigma fixed at 0 for sparsity analysis
SIGMA = 0.0

# Test samples limit (for speed - need enough samples for meaningful sparsity)
MAX_TEST_SAMPLES = 100

# CIM parameters (fixed for all experiments)
QUANT_MODE = 'CIM_Quant'
WEIGHT_BITS = 8
INPUT_BITS = 8
ADC_BITS = 6
PARALLEL_READ = 64
USE_PARTIAL_SUM_QUANT = False
ENABLE_ACTIVITY_STATS = False  # Disable activity counting for sparsity sweep
ENABLE_SPARSITY_STATS = True  # Enable sparsity stats
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
NUM_WORKERS = 4
BATCH_SIZE = 64


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_test_loader(max_samples=MAX_TEST_SAMPLES):
    """Create test data loader with limited samples."""
    test_tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_dataset = datasets.ImageFolder(root=cfg.DATASET_DIR, transform=test_tf)

    # Limit dataset size
    if len(test_dataset) > max_samples:
        # Create a subset
        indices = list(range(max_samples))
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == 'cuda')
    )

    return test_loader


def run_single_experiment(
    weight_encode,
    activation_encode,
    pruning_rate,
    sigma,
    base_state_dict,
    test_loader,
    exp_id
):
    """Run a single experiment with given parameters."""

    # Build model for current configuration
    model = build_model_for_current_mode(
        sigma=sigma,
        base_state_dict=base_state_dict,
        quant_mode=QUANT_MODE,
        weight_encode_method=weight_encode,
        activation_encode_method=activation_encode,
        weight_bits=WEIGHT_BITS,
        input_bits=INPUT_BITS,
        adc_bits=ADC_BITS,
        parallel_read=PARALLEL_READ,
        use_partial_sum_quant=USE_PARTIAL_SUM_QUANT,
        noise_enable=False,
        noise_mode='exclude',
        include_layers=[],
        exclude_layers=['patch_embed.proj', 'head'],
        device=DEVICE,
        num_classes=NUM_CLASSES,
        enable_activity_stats=ENABLE_ACTIVITY_STATS,
        enable_sparsity_stats=ENABLE_SPARSITY_STATS
    )

    # Run inference to collect sparsity stats
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with maybe_autocast(DEVICE):
                outputs = model(images)

            # Only need one batch for sparsity stats
            break

    # Collect sparsity statistics
    sparsity_stats = collect_sparsity_stats(model)

    # Compute average densities
    avg_weight_density = 'N/A'
    avg_activation_density = 'N/A'

    if sparsity_stats['weight_density']:
        all_w_density = []
        for ws in sparsity_stats['weight_density']:
            all_w_density.extend([d for _, d in ws['data']])
        avg_weight_density = sum(all_w_density) / len(all_w_density) if all_w_density else 'N/A'

    if sparsity_stats['activation_density']:
        all_a_density = []
        for as_ in sparsity_stats['activation_density']:
            all_a_density.extend([d for _, d in as_['data']])
        avg_activation_density = sum(all_a_density) / len(all_a_density) if all_a_density else 'N/A'

    # Print detailed per-layer sparsity info
    print(f"\n[Exp {exp_id}] Config: {QUANT_MODE}_W{weight_encode[0]}_A{activation_encode[0]}_P{pruning_rate}_S{sigma}")

    if sparsity_stats['weight_density']:
        print(f"  Weight Density (per bit, avg={avg_weight_density:.3f}):")
        for ws in sparsity_stats['weight_density']:
            data = ws['data']
            # If even number of bits (differential encoding), split into + and -
            if len(data) % 2 == 0:
                half = len(data) // 2
                pos_bits = [f"bit{i}+={d:.3f}" for i, (_, d) in enumerate(data[:half])]
                neg_bits = [f"bit{i}-={d:.3f}" for i, (_, d) in enumerate(data[half:])]
                print(f"    {ws['layer']}: " + ", ".join(pos_bits + neg_bits))
            else:
                print(f"    {ws['layer']}: " + ", ".join([f"bit{b}={d:.3f}" for b, d in data]))

    if sparsity_stats['activation_density']:
        print(f"  Activation Density (per bit, avg={avg_activation_density:.3f}):")
        for as_ in sparsity_stats['activation_density']:
            data = as_['data']
            # If even number of bits (differential encoding), split into + and -
            if len(data) % 2 == 0:
                half = len(data) // 2
                pos_bits = [f"bit{i}+={d:.3f}" for i, (_, d) in enumerate(data[:half])]
                neg_bits = [f"bit{i}-={d:.3f}" for i, (_, d) in enumerate(data[half:])]
                print(f"    {as_['layer']}: " + ", ".join(pos_bits + neg_bits))
            else:
                print(f"    {as_['layer']}: " + ", ".join([f"bit{b}={d:.3f}" for b, d in data]))

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'avg_weight_density': avg_weight_density,
        'avg_activation_density': avg_activation_density,
        'weight_density_per_layer': sparsity_stats['weight_density'],
        'activation_density_per_layer': sparsity_stats['activation_density']
    }


def prepare_pruned_state_dict(pruning_rate, seed=2026):
    """Prepare base state dict with pruning applied. Returns (state_dict, pruning_stats)."""
    model_loader_fn = lambda: load_model_from_checkpoint(
        cfg.MODEL_PATH,
        num_classes=NUM_CLASSES
    )

    base_state_dict, pruning_stats = prepare_base_state_dict_for_run(
        model_loader_fn,
        pruning_enable=(pruning_rate > 0),
        pruning_rate=pruning_rate,
        exclude_keywords=['patch_embed', 'head']
    )

    return base_state_dict, pruning_stats


# ==============================================================================
# Main Sweep Function
# ==============================================================================

def run_sweep_experiment():
    """Run the complete sweep experiment."""

    # Output directory (Results subfolder)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Info] Output directory: {output_dir}")
    print(f"[Info] Device: {DEVICE}")
    print(f"[Info] Total experiments: {len(ENCODING_COMBOS) * len(PRUNING_RATES)}")

    # Set random seed
    setup_seed(2026)

    # Create test loader (shared across experiments)
    print("[Info] Creating test loader...")
    test_loader = create_test_loader(MAX_TEST_SAMPLES)
    print(f"[Info] Test samples: {len(test_loader.dataset)}")

    # Results storage
    results = []

    # Total experiments
    total_experiments = len(ENCODING_COMBOS) * len(PRUNING_RATES)
    exp_counter = 0

    # Start timing
    start_time = time.time()

    # Pre-generate base state dicts for each pruning rate (to avoid reloading)
    print("\n[Info] Preparing base state dicts for each pruning rate...")
    state_dicts = {}
    for pr in tqdm(PRUNING_RATES, desc="Loading pruned models"):
        state_dicts[pr], pruning_stats = prepare_pruned_state_dict(pr)
        if pruning_stats is not None:
            sparsity = pruning_stats['sparsity']
        else:
            sparsity = 0.0
        print(f"  Pruning rate {pr}: sparsity = {sparsity:.4f}")

    # Main sweep loop
    print("\n" + "=" * 80)
    print("Starting Main Sweep")
    print("=" * 80)

    for weight_encode, activation_encode in ENCODING_COMBOS:
        for pruning_rate in PRUNING_RATES:
            exp_counter += 1

            # Get the appropriate base state dict
            base_state_dict = state_dicts[pruning_rate]

            # Run experiment
            sparsity_result = run_single_experiment(
                weight_encode=weight_encode,
                activation_encode=activation_encode,
                pruning_rate=pruning_rate,
                sigma=SIGMA,
                base_state_dict=base_state_dict,
                test_loader=test_loader,
                exp_id=exp_counter
            )

            # Store result
            result = {
                'exp_id': exp_counter,
                'weight_encode': weight_encode,
                'activation_encode': activation_encode,
                'pruning_rate': pruning_rate,
                'sigma': SIGMA,
                'avg_weight_density': sparsity_result['avg_weight_density'],
                'avg_activation_density': sparsity_result['avg_activation_density'],
            }
            results.append(result)

            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / exp_counter
            remaining = avg_time * (total_experiments - exp_counter)

            print(
                f"[{exp_counter:2d}/{total_experiments}] "
                f"W={weight_encode[:4]:<4} A={activation_encode[:4]:<4} "
                f"P={pruning_rate:.1f} "
                f"-> W_density={sparsity_result['avg_weight_density']:.3f} "
                f"A_density={sparsity_result['avg_activation_density']:.3f} | "
                f"Elapsed: {elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min"
            )

    # Total time
    total_time = time.time() - start_time
    print(f"\n[Info] Total experiment time: {total_time/60:.1f} minutes")

    # Save results to CSV
    df = pd.DataFrame(results)

    # Save to CSV
    output_file = os.path.join(output_dir, "sparsity_results.csv")

    df.to_csv(output_file, index=False)
    print(f"\n[Info] Results saved to: {output_file}")

    # Also print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    run_sweep_experiment()
