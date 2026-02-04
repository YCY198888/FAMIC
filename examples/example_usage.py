"""
Example usage of FAMIC model evaluation.

This script demonstrates how to:
1. Create embedding matrix
2. Load the FAMIC model with pretrained weights
3. Load datasets
4. Run evaluation
"""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import FAMIC, create_embedding_matrix, EMBEDDING_DIMENSIONS, VOCAB_LENGTH, NUM_HEADS
from src.datasets import load_dataset, get_dataset_info
from src.evaluate import evaluate_model, plot_confusion_matrix


def evaluate_dataset(dataset_name: str, device: torch.device, embedding_matrix: torch.Tensor) -> dict:
    """Evaluate model on a single dataset."""
    print("\n" + "="*70)
    print(f"Evaluating on '{dataset_name}' dataset")
    print("="*70)
    
    # Step 1: Load model with pretrained weights
    print(f"\nStep 1: Loading model with pretrained weights for '{dataset_name}'...")
    try:
        model = FAMIC.from_pretrained_huggingface(
            dataset_name=dataset_name,
            embedding_matrix=embedding_matrix,
            cache_dir="models",
            version="v2",
            device=device,
            hidden_dim=EMBEDDING_DIMENSIONS,
            n_layers=2,
            max_relative_position_mask=2,
            max_relative_position_shift=3,
            pivot=0.5,
            num_heads=NUM_HEADS,
            drop_prob=0.5,
            digits_dim=1
        )
        print("✓ Model loaded with pretrained weights\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}\n")
        print("Please check that:")
        print("  1. You have internet connection to download weights from HuggingFace")
        print("  2. The dataset name is correct ('twitter' or 'wine')")
        return None
    
    # Step 2: Load dataset
    print(f"Step 2: Loading '{dataset_name}' dataset...")
    
    # Get dataset info
    try:
        dataset_info = get_dataset_info(dataset_name)
        print(f"  Display name: {dataset_info.get('display_name', 'N/A')}")
        print(f"  Total rows: {dataset_info.get('total_rows', 'N/A'):,}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not get dataset info: {e}")
    
    # Load test dataset
    try:
        test_loader = load_dataset(
            dataset_name,
            data_root="data",
            split="test",
            batch_size=32,
            shuffle=False
        )
        print("✓ Dataset loaded\n")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}\n")
        print("Please check that:")
        print("  1. You have internet connection to download dataset from HuggingFace")
        print("  2. The dataset name is correct ('twitter' or 'wine')")
        return None
    
    # Step 3: Evaluate model
    print(f"Step 3: Evaluating model on '{dataset_name}'...")
    try:
        metrics = evaluate_model(
            model,
            test_loader,
            device=device,
            use_mask=True,
            use_shift1=True,
            use_shift2=True,
            class_names=["Negative", "Positive"]
        )
        print("✓ Evaluation complete\n")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}\n")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 4: Visualize confusion matrix
    print(f"Step 4: Saving confusion matrix for '{dataset_name}'...")
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        plot_confusion_matrix(
            confusion_matrix=metrics['confusion_matrix'],
            class_names=["Negative", "Positive"],
            save_path=f"results/confusion_matrix_{dataset_name}.png",
            title=f"Confusion Matrix - {dataset_name}"
        )
        print("✓ Confusion matrix saved\n")
    except Exception as e:
        print(f"⚠ Warning: Could not save confusion matrix: {e}\n")
    
    return metrics


def main():
    """Example usage of FAMIC evaluation on both datasets."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("FAMIC Model Evaluation - Both Datasets")
    print("="*70)
    print(f"Using device: {device}\n")
    
    # Step 1: Create embedding matrix (shared for both datasets)
    print("Step 1: Creating embedding matrix...")
    embedding_matrix = create_embedding_matrix(
        vocab_length=VOCAB_LENGTH,
        embedding_dim=EMBEDDING_DIMENSIONS
    )
    print(f"✓ Embedding matrix created: {embedding_matrix.shape}\n")
    
    # Evaluate on both datasets
    datasets = ["wine", "twitter"]
    all_results = {}
    
    for dataset_name in datasets:
        metrics = evaluate_dataset(dataset_name, device, embedding_matrix)
        if metrics:
            all_results[dataset_name] = metrics
    
    # Print final summary
    print("\n" + "="*70)
    print("Final Evaluation Summary")
    print("="*70)
    
    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'loss' in metrics:
            print(f"  Loss:      {metrics['loss']:.4f}")
    
    print("\n" + "="*70)
    print("All evaluations complete!")
    print("="*70)


if __name__ == "__main__":
    main()

