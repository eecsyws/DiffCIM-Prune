"""
Sweep Experiment Script for DiffCIM-Prune

This script runs a comprehensive sweep over:
- Weight encoding: twos_complement / differential
- Activation encoding: twos_complement / differential
- Pruning rate: 0, 0.1, 0.2, 0.3, 0.4
- Variation sigma: 0, 0.05, 0.1, 0.15, 0.2

Total: 4 x 5 x 5 = 100 experiments
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
from src.inference import setup_seed, maybe_autocast, CIFAR100_MEAN, CIFAR100_STD
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

# Variation sigma values
SIGMA_LIST = [0, 0.05, 0.1, 0.15, 0.2]

# Pruning rates
PRUNING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# Test samples limit (for speed)
MAX_TEST_SAMPLES = 1000

# CIM parameters (fixed for all experiments)
QUANT_MODE = 'CIM_Quant'
WEIGHT_BITS = 8
INPUT_BITS = 8
ADC_BITS = 6
PARALLEL_READ = 64
USE_PARTIAL_SUM_QUANT = False
ENABLE_ACTIVITY_STATS = False  # Set to True to enable 1x1 interaction counting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
NUM_WORKERS = 4  # Reduce for faster loading
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
    test_loader
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
        enable_activity_stats=ENABLE_ACTIVITY_STATS
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with maybe_autocast(DEVICE):
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if total >= MAX_TEST_SAMPLES:
                break

    accuracy = 100.0 * correct / total

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return accuracy


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

    # Output directory (same folder as this script)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Info] Output directory: {output_dir}")
    print(f"[Info] Device: {DEVICE}")
    print(f"[Info] Total experiments: {len(ENCODING_COMBOS) * len(SIGMA_LIST) * len(PRUNING_RATES)}")

    # Set random seed
    setup_seed(2026)

    # Create test loader (shared across experiments)
    print("[Info] Creating test loader...")
    test_loader = create_test_loader(MAX_TEST_SAMPLES)
    print(f"[Info] Test samples: {len(test_loader.dataset)}")

    # Results storage
    results = []

    # Total experiments
    total_experiments = len(ENCODING_COMBOS) * len(SIGMA_LIST) * len(PRUNING_RATES)
    exp_counter = 0

    # Start timing
    start_time = time.time()

    # Pre-generate base state dicts for each pruning rate (to avoid reloading)
    print("\n[Info] Preparing base state dicts for each pruning rate...")
    state_dicts = {}
    for pr in tqdm(PRUNING_RATES, desc="Loading pruned models"):
        state_dicts[pr], pruning_stats = prepare_pruned_state_dict(pr)
        # Get pruning stats (sparsity is 0 when pruning is disabled)
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
            for sigma in SIGMA_LIST:
                exp_counter += 1

                # Get the appropriate base state dict
                base_state_dict = state_dicts[pruning_rate]

                # Run experiment
                accuracy = run_single_experiment(
                    weight_encode=weight_encode,
                    activation_encode=activation_encode,
                    pruning_rate=pruning_rate,
                    sigma=sigma,
                    base_state_dict=base_state_dict,
                    test_loader=test_loader
                )

                # Store result
                result = {
                    'exp_id': exp_counter,
                    'weight_encode': weight_encode,
                    'activation_encode': activation_encode,
                    'pruning_rate': pruning_rate,
                    'sigma': sigma,
                    'accuracy': accuracy,
                }
                results.append(result)

                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / exp_counter
                remaining = avg_time * (total_experiments - exp_counter)

                print(
                    f"[{exp_counter:3d}/{total_experiments}] "
                    f"W={weight_encode[:4]:<4} A={activation_encode[:4]:<4} "
                    f"P={pruning_rate:.1f} σ={sigma:.2f} "
                    f"-> Acc={accuracy:.2f}% | "
                    f"Elapsed: {elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min"
                )

    # Total time
    total_time = time.time() - start_time
    print(f"\n[Info] Total experiment time: {total_time/60:.1f} minutes")

    # Save results to CSV only
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Info] Results saved to: {csv_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("Final Results Summary")
    print("=" * 80)

    # Pivot table: sigma x encoding
    print("\n--- Accuracy by Sigma and Encoding (Pruning Rate = 0.0) ---")
    pivot = df[df['pruning_rate'] == 0.0].pivot_table(
        values='accuracy',
        index=['weight_encode', 'activation_encode'],
        columns='sigma',
        aggfunc='first'
    )
    print(pivot.to_string())

    print("\n--- Accuracy by Pruning Rate (sigma = 0.2, differential encoding) ---")
    pivot2 = df[
        (df['sigma'] == 0.2) &
        (df['weight_encode'] == 'differential') &
        (df['activation_encode'] == 'differential')
    ].pivot_table(
        values='accuracy',
        index='pruning_rate',
        columns='sigma',
        aggfunc='first'
    )
    print(pivot2.to_string())

    return csv_path


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    csv_path = run_sweep_experiment()
    print(f"\n[Done] Results saved to: {csv_path}")
