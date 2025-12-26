"""
Example usage of FAMIC model evaluation.

This script demonstrates how to:
1. Load the FAMIC model
2. Load datasets
3. Run evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import FAMIC
from src.download_weights import get_weights_path
from src.datasets import load_dataset, get_dataset_info
from src.evaluate import evaluate_model, plot_confusion_matrix


def main():
    """Example usage of FAMIC evaluation."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Step 1: Load model weights
    print("Step 1: Loading model weights...")
    try:
        weights_path = get_weights_path()
        print(f"✓ Weights loaded from: {weights_path}\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}\n")
        print("Please update src/download_weights.py with the actual download URL")
        return
    
    # Step 2: Initialize model
    print("Step 2: Initializing model...")
    # TODO: Update these parameters based on your model
    model = FAMIC.from_pretrained(
        weights_path,
        device=device,
        input_dim=512,  # TODO: Update
        hidden_dim=256,  # TODO: Update
        num_classes=2   # TODO: Update
    )
    print("✓ Model initialized\n")
    
    # Step 3: Load dataset
    print("Step 3: Loading dataset...")
    dataset_name = "dataset1"  # or "dataset2"
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Input dim: {dataset_info['input_dim']}\n")
    
    # Load test dataset
    test_loader = load_dataset(
        dataset_name,
        data_root="data",
        split="test",
        batch_size=32,
        shuffle=False
    )
    print("✓ Dataset loaded\n")
    
    # Step 4: Evaluate model
    print("Step 4: Evaluating model...")
    metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        class_names=None  # TODO: Add class names if available
    )
    
    # Step 5: Visualize confusion matrix
    print("Step 5: Saving confusion matrix...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path="results/example_confusion_matrix.png",
        title=f"Confusion Matrix - {dataset_name}"
    )
    
    print("\n✓ Evaluation complete!")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()

