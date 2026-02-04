#!/usr/bin/env python
"""
Main evaluation script for FAMIC model.

This script:
1. Creates embedding matrix
2. Loads the FAMIC model with pretrained weights
3. Loads test datasets
4. Evaluates model performance
5. Reports metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.model import FAMIC, create_embedding_matrix, EMBEDDING_DIMENSIONS, VOCAB_LENGTH, NUM_HEADS
from src.evaluate import evaluate_on_datasets


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    print("="*70)
    print("FAMIC Model Evaluation")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Set device
    if config['evaluation']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['evaluation']['device'])
    
    print(f"Using device: {device}\n")
    
    # Step 1: Create embedding matrix
    print("Step 1: Creating embedding matrix...")
    embedding_matrix = create_embedding_matrix(
        vocab_length=config['model'].get('vocab_length', VOCAB_LENGTH),
        embedding_dim=config['model'].get('embedding_dim', EMBEDDING_DIMENSIONS)
    )
    print(f"✓ Embedding matrix created: {embedding_matrix.shape}\n")
    
    # Step 2: Get dataset names from config
    dataset_names = []
    for key in config['datasets'].keys():
        if key not in ['data_root', 'default_dataset'] and isinstance(config['datasets'][key], dict):
            dataset_names.append(key)
    
    if not dataset_names:
        dataset_names = [config['datasets'].get('default_dataset', 'wine')]
    
    print(f"Step 2: Evaluating on {len(dataset_names)} dataset(s): {', '.join(dataset_names)}\n")
    
    # Step 3: Load model and evaluate for each dataset
    results = {}
    for dataset_name in dataset_names:
        print("="*70)
        print(f"Evaluating on '{dataset_name}' dataset")
        print("="*70)
        
        # Load model with pretrained weights for this dataset
        print(f"\nLoading model with pretrained weights for '{dataset_name}'...")
        try:
            model = FAMIC.from_pretrained_huggingface(
                dataset_name=dataset_name,
                embedding_matrix=embedding_matrix,
                cache_dir=config['weights']['save_dir'],
                version=config['weights']['version'],
                device=device,
                hidden_dim=config['model'].get('hidden_dim', EMBEDDING_DIMENSIONS),
                n_layers=config['model'].get('n_layers', 2),
                max_relative_position_mask=config['model'].get('max_relative_position_mask', 2),
                max_relative_position_shift=config['model'].get('max_relative_position_shift', 3),
                pivot=config['model'].get('pivot', 0.5),
                num_heads=config['model'].get('num_heads', NUM_HEADS),
                drop_prob=config['model'].get('drop_prob', 0.5),
                digits_dim=config['model'].get('digits_dim', 1)
            )
            print("✓ Model loaded successfully\n")
        except Exception as e:
            print(f"✗ Error loading model: {e}\n")
            print("Please check that:")
            print("  1. You have internet connection to download weights from HuggingFace")
            print("  2. The dataset name is correct ('twitter' or 'wine')")
            continue
        
        # Evaluate on dataset
        print(f"Evaluating on '{dataset_name}' dataset...")
        try:
            dataset_results = evaluate_on_datasets(
                model,
                dataset_names=[dataset_name],
                data_root=config['datasets']['data_root'],
                device=device,
                batch_size=config['evaluation']['batch_size'],
                use_mask=config['evaluation'].get('use_mask', True),
                use_shift1=config['evaluation'].get('use_shift1', True),
                use_shift2=config['evaluation'].get('use_shift2', True),
                save_results=config['evaluation']['save_results'],
                results_dir=config['evaluation']['results_dir']
            )
            results[dataset_name] = dataset_results.get(dataset_name, {})
            print(f"✓ Evaluation complete for '{dataset_name}'\n")
        except Exception as e:
            print(f"✗ Error during evaluation: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("="*70)
    print("Evaluation Summary")
    print("="*70)
    for dataset_name, dataset_results in results.items():
        if dataset_results:
            print(f"\n{dataset_name}:")
            print(f"  Accuracy:  {dataset_results.get('accuracy', 'N/A'):.4f}" if isinstance(dataset_results.get('accuracy'), float) else f"  Accuracy:  {dataset_results.get('accuracy', 'N/A')}")
            print(f"  Precision: {dataset_results.get('precision', 'N/A'):.4f}" if isinstance(dataset_results.get('precision'), float) else f"  Precision: {dataset_results.get('precision', 'N/A')}")
            print(f"  Recall:    {dataset_results.get('recall', 'N/A'):.4f}" if isinstance(dataset_results.get('recall'), float) else f"  Recall:    {dataset_results.get('recall', 'N/A')}")
            print(f"  F1-Score:  {dataset_results.get('f1_score', 'N/A'):.4f}" if isinstance(dataset_results.get('f1_score'), float) else f"  F1-Score:  {dataset_results.get('f1_score', 'N/A')}")
    
    if config['evaluation']['save_results']:
        print(f"\nResults saved to {config['evaluation']['results_dir']}/")
    print("="*70)


if __name__ == "__main__":
    main()

