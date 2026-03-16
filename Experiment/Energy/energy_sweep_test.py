"""
Energy Sweep Experiment Script for DiffCIM-Prune

This script runs a comprehensive sweep over:
- Weight encoding: twos_complement / differential
- Activation encoding: twos_complement / differential
- Pruning rate: 0, 0.1, 0.2, 0.3, 0.4
- Variation sigma: 0, 0.05, 0.1, 0.15, 0.2

Total: 4 x 5 x 5 = 100 experiments

Output: Activity statistics (1x1 interactions, activation 1s, weight 1s)
"""

import os
import sys
import gc
import time

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

# Fixed sigma (no variation for energy measurement)
SIGMA = 0.0

# Pruning rates
PRUNING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# Test samples limit (1 sample for energy measurement)
MAX_TEST_SAMPLES = 1

# CIM parameters (fixed for all experiments)
QUANT_MODE = 'CIM_Quant'
WEIGHT_BITS = 8
INPUT_BITS = 8
ADC_BITS = 6
PARALLEL_READ = 64
USE_PARTIAL_SUM_QUANT = False
ENABLE_ACTIVITY_STATS = True  # Enable activity counting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
NUM_WORKERS = 2
BATCH_SIZE = 1  # Single sample for energy measurement
SEED = 2026


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

    # Limit dataset size to 1 sample
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


def collect_activity_stats(model):
    """Collect activity statistics from CIM layers."""
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

    # Run inference on 1 sample
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with maybe_autocast(DEVICE):
                outputs = model(images)

    # Collect activity statistics
    activity_stats = collect_activity_stats(model)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activity_stats


def prepare_pruned_state_dict(pruning_rate):
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

    # Output directory: Energy/Results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Info] Output directory: {output_dir}")
    print(f"[Info] Device: {DEVICE}")
    print(f"[Info] Total experiments: {len(ENCODING_COMBOS) * len(PRUNING_RATES)}")

    # Set random seed
    setup_seed(SEED)

    # Create test loader (1 sample)
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

    # Pre-generate base state dicts for each pruning rate
    print("\n[Info] Preparing base state dicts for each pruning rate...")
    state_dicts = {}
    for pr in tqdm(PRUNING_RATES, desc="Loading pruned models"):
        state_dicts[pr], _ = prepare_pruned_state_dict(pr)

    # Main sweep loop
    print("\n" + "=" * 80)
    print("Starting Energy Sweep (Activity Statistics)")
    print("=" * 80)

    for weight_encode, activation_encode in ENCODING_COMBOS:
        for pruning_rate in PRUNING_RATES:
            exp_counter += 1

            # Get the appropriate base state dict
            base_state_dict = state_dicts[pruning_rate]

            # Run experiment
            activity_stats = run_single_experiment(
                weight_encode=weight_encode,
                activation_encode=activation_encode,
                pruning_rate=pruning_rate,
                sigma=SIGMA,
                base_state_dict=base_state_dict,
                test_loader=test_loader
            )

            # Store result
            result = {
                'exp_id': exp_counter,
                'weight_encode': weight_encode,
                'activation_encode': activation_encode,
                'pruning_rate': pruning_rate,
                'sigma': SIGMA,
                'total_1x1': activity_stats['total_1x1_count'],
                'total_act_1': activity_stats['total_1_count'],
                'total_weight_1': activity_stats['total_weight_1_count'],
            }
            results.append(result)

            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / exp_counter
            remaining = avg_time * (total_experiments - exp_counter)

            print(
                f"[{exp_counter:3d}/{total_experiments}] "
                f"W={weight_encode[:4]:<4} A={activation_encode[:4]:<4} "
                f"P={pruning_rate:.1f} "
                f"-> 1x1={activity_stats['total_1x1_count']:.0f} | "
                f"Act1={activity_stats['total_1_count']:.0f} | "
                f"Wt1={activity_stats['total_weight_1_count']:.0f} | "
                f"Elapsed: {elapsed/60:.1f}min"
            )

    # Total time
    total_time = time.time() - start_time
    print(f"\n[Info] Total experiment time: {total_time/60:.1f} minutes")

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "energy_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Info] Results saved to: {csv_path}")

    return csv_path


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    csv_path = run_sweep_experiment()
    print(f"\n[Done] Results saved to: {csv_path}")
