"""
Model Evaluation Script

This module provides utilities for evaluating FAMIC model performance.
Reports: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.model import FAMIC


def evaluate_model(
    model: FAMIC,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute metrics.
    
    Args:
        model: Trained FAMIC model
        dataloader: DataLoader for the test dataset
        device: Device to run evaluation on (default: cuda if available, else cpu)
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print classification report if class names provided
    if class_names:
        print("\nClassification Report:")
        print(classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        ))
    
    print("="*50 + "\n")
    
    return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Optional list of class names
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def evaluate_on_datasets(
    model: FAMIC,
    dataset_names: list,
    data_root: str = "data",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple datasets.
    
    Args:
        model: Trained FAMIC model
        dataset_names: List of dataset names to evaluate on
        data_root: Root directory containing datasets
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        save_results: Whether to save results to files
        results_dir: Directory to save results
        
    Returns:
        Dictionary mapping dataset names to their evaluation metrics
    """
    from src.datasets import load_dataset
    
    all_results = {}
    
    for dataset_name in dataset_names:
        print(f"\nEvaluating on {dataset_name}...")
        print("-" * 50)
        
        # Load test dataset
        test_loader = load_dataset(
            dataset_name,
            data_root=data_root,
            split="test",
            batch_size=batch_size,
            shuffle=False
        )
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device=device)
        
        all_results[dataset_name] = metrics
        
        # Save confusion matrix plot
        if save_results:
            cm_path = Path(results_dir) / f"{dataset_name}_confusion_matrix.png"
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=str(cm_path),
                title=f"Confusion Matrix - {dataset_name}"
            )
    
    # Save summary results
    if save_results:
        summary_path = Path(results_dir) / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("FAMIC Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, metrics in all_results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write("\n")
        
        print(f"\nResults saved to {results_dir}/")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("Evaluation script")
    print("TODO: Load model and datasets, then run evaluation")
    
    # Example:
    # from src.model import FAMIC
    # from src.download_weights import get_weights_path
    # 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # weights_path = get_weights_path()
    # model = FAMIC.from_pretrained(weights_path, device=device)
    # 
    # results = evaluate_on_datasets(
    #     model,
    #     dataset_names=["dataset1", "dataset2"],
    #     device=device
    # )

