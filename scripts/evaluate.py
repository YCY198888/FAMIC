#!/usr/bin/env python
"""
Main evaluation script for FAMIC model.

This script:
1. Loads the FAMIC model with pretrained weights
2. Loads test datasets
3. Evaluates model performance
4. Reports metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.model import FAMIC
from src.download_weights import get_weights_path
from src.evaluate import evaluate_on_datasets


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    print("="*60)
    print("FAMIC Model Evaluation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Set device
    if config['evaluation']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['evaluation']['device'])
    
    print(f"Using device: {device}\n")
    
    # Load model weights
    print("Loading model weights...")
    try:
        weights_path = get_weights_path(
            weights_dir=config['weights']['save_dir'],
            weights_filename=config['weights']['filename']
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease:")
        print("1. Download model weights manually, or")
        print("2. Update the download URL in src/download_weights.py")
        return
    
    # Initialize model
    print("Initializing model...")
    model = FAMIC.from_pretrained(
        weights_path,
        device=device,
        **config['model']
    )
    print("Model loaded successfully.\n")
    
    # Get dataset names from config
    dataset_names = list(config['datasets'].keys())
    # Remove 'data_root' if it's in the keys
    if 'data_root' in dataset_names:
        dataset_names.remove('data_root')
    
    # Evaluate on datasets
    print(f"Evaluating on {len(dataset_names)} dataset(s)...")
    results = evaluate_on_datasets(
        model,
        dataset_names=dataset_names,
        data_root=config['datasets']['data_root'],
        device=device,
        batch_size=config['evaluation']['batch_size'],
        save_results=config['evaluation']['save_results'],
        results_dir=config['evaluation']['results_dir']
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to {config['evaluation']['results_dir']}/")


if __name__ == "__main__":
    main()

