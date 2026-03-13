# ==============================================================================
# Main Entry Point for Differential Encode Project
# ==============================================================================
# This file serves as the main entry point for running the quantized inference.
# All configuration can be modified in config/config.py
#
# Usage:
#   python main.py
# ==============================================================================

from src.inference import run_inference

if __name__ == '__main__':
    run_inference()
