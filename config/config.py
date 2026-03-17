# ==============================================================================
# Configuration File for Differential Encode Project
# ==============================================================================
# This file contains all configurable parameters for the project.
# Users can modify this file to control experiment settings.
# ==============================================================================

import torch

# ==============================================================================
# Section 1: Path Settings
# ==============================================================================

import os


MODEL_PATH = '/lustre/home/2200012654/model/timm/vit_tiny_cifar100/vit_tiny_cifar100_finetune.pt'
DATASET_DIR = '/lustre/home/2200012654/dataset/cifar100/test'

# ==============================================================================
# Section 2: Runtime Settings
# ==============================================================================

BATCH_SIZE = 128
NUM_WORKERS = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 100
SEED = 2026
MAX_TEST_SAMPLES = 10000

# ==============================================================================
# Section 3: Quantization Mode Settings
# ==============================================================================

# quant_mode:
#   'CIM_Quant' -> use CIM-based layer replacement
#   'Fake_Quant' -> use simplified fake-quant/noisy layer replacement
QUANT_MODE = 'CIM_Quant'

# Weight encoding method:
#   'twos_complement' -> single-array / signed style (traditional)
#   'differential' -> differential style (proposed)
WEIGHT_ENCODE_METHOD = 'differential'

# Activation encoding method:
#   'twos_complement' -> single-array / signed style (traditional)
#   'differential' -> differential style (proposed)
ACTIVATION_ENCODE_METHOD = 'differential'

# Shared quantization bit-width settings
WEIGHT_BITS = 8
INPUT_BITS = 8

# CIM_Quant-specific settings
ADC_BITS = 6
PARALLEL_READ = 64
USE_PARTIAL_SUM_QUANT = False

# Fake_quant-specific settings
NOISE_ENABLE = True
NOISE_MODE = 'exclude'
INCLUDE_LAYERS = []
EXCLUDE_LAYERS = ['patch_embed.proj', 'head']

# Sweep only sigma. ADC and parallel_read are fixed by config.
VARIATION_SIGMA_LIST = [0, 0.05, 0.1, 0.15, 0.2]

# ==============================================================================
# Section 4: Pruning Settings
# ==============================================================================

# Global unstructured pruning is applied BEFORE any quant-layer replacement.
# Target:
#   - only nn.Linear.weight
#   - exclude layers whose names contain 'patch_embed' or 'head'
PRUNING_ENABLE = False
PRUNING_RATE = 0.0
PRUNING_EXCLUDE_KEYWORDS = ['patch_embed', 'head']

# ==============================================================================
# Section 4b: Activity Statistics Settings
# ==============================================================================

# Enable activity statistics (count 1x1 interactions for energy analysis)
ENABLE_ACTIVITY_STATS = False

# Enable per-bit-plane sparsity statistics (weight & activation)
ENABLE_SPARSITY_STATS = False

# ==============================================================================
# Section 5: Dataset Normalization Settings
# ==============================================================================

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ==============================================================================
# Export all config variables for easy import
# ==============================================================================

__all__ = [
    # Path settings
    'MODEL_PATH',
    'DATASET_DIR',
    # Runtime settings
    'BATCH_SIZE',
    'NUM_WORKERS',
    'DEVICE',
    'NUM_CLASSES',
    'SEED',
    'MAX_TEST_SAMPLES',
    # Quantization settings
    'QUANT_MODE',
    'WEIGHT_ENCODE_METHOD',
    'ACTIVATION_ENCODE_METHOD',
    'WEIGHT_BITS',
    'INPUT_BITS',
    'ADC_BITS',
    'PARALLEL_READ',
    'USE_PARTIAL_SUM_QUANT',
    'NOISE_ENABLE',
    'NOISE_MODE',
    'INCLUDE_LAYERS',
    'EXCLUDE_LAYERS',
    'VARIATION_SIGMA_LIST',
    # Pruning settings
    'PRUNING_ENABLE',
    'PRUNING_RATE',
    'PRUNING_EXCLUDE_KEYWORDS',
    # Activity statistics settings
    'ENABLE_ACTIVITY_STATS',
    'ENABLE_SPARSITY_STATS',
    # Dataset settings
    'CIFAR100_MEAN',
    'CIFAR100_STD',
]