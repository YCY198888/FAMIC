"""
Model Evaluation Script

This module provides utilities for evaluating FAMIC model performance.
Reports: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
"""

import torch
import torch.nn as nn
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
    criterion: Optional[nn.Module] = None,
    use_mask: bool = False,
    use_shift1: bool = False,
    use_shift2: bool = False,
    class_names: Optional[list] = None,
    return_loss: bool = True
) -> Dict[str, float]:
    """
    Evaluate FAMIC model on a dataset and compute metrics.
    
    This function evaluates the model using the same approach as the original training code:
    - Uses BCELoss for binary classification
    - Applies sigmoid to outputs and thresholds at 0.5
    - Computes accuracy, precision, recall, F1, and confusion matrix
    
    Args:
        model: Trained FAMIC model
        dataloader: DataLoader for the test dataset (should return dict with 'word_ids', 'attention_mask', 'labels')
        device: Device to run evaluation on (default: cuda if available, else cpu)
        criterion: Loss function (default: nn.BCEWithLogitsLoss())
        use_mask: Whether to use learned mask in forward pass
        use_shift1: Whether to use first shifter in forward pass
        use_shift2: Whether to use second shifter in forward pass
        class_names: Optional list of class names for reporting
        return_loss: Whether to return average loss (default: True)
        
    Returns:
        Dictionary containing evaluation metrics and optionally loss
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    losses = []
    masks_val = []
    
    num_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract batch data
            word_ids = batch["word_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Convert to long (token indices)
            word_ids = word_ids.long()
            
            # Forward pass through model
            outputs = model(
                word_ids,
                use_mask=use_mask,
                use_shift1=use_shift1,
                use_shift2=use_shift2
            )
            
            # Get model output (before sigmoid)
            model_output = outputs['output']  # (B,)
            mask_out = outputs['mask']  # (B, L)
            
            # Compute loss
            loss = criterion(model_output.squeeze(), labels.float())
            losses.append(loss.item())
            
            # Get predictions: apply sigmoid and threshold at 0.5
            pred = torch.round((torch.sigmoid(model_output.squeeze()) >= 0.5).long())
            
            # Calculate accuracy
            correct_tensor = pred.eq(labels.float().view_as(pred))
            num_correct += correct_tensor.sum().item()
            total_samples += labels.size(0)
            
            # Calculate mask statistics (mean mask value per sample)
            # Sum of mask values divided by number of tokens (using attention_mask)
            word_count = attention_mask.sum(dim=1).clamp_min(1)  # (B,)
            mask_sum = mask_out.sum(dim=1)  # (B,)
            masks_val.append((mask_sum / word_count).mean().item())
            
            # Store predictions and labels
            all_predictions.extend(pred.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions).astype(int).reshape(-1)
    y_true = np.array(all_labels).astype(int).reshape(-1)
    
    # Calculate confusion matrix components
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calculate metrics
    acc = (TP + TN) / (TP + FP + TN + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    # Average loss
    avg_loss = np.mean(losses) if return_loss else None
    
    # Average mask value
    avg_mask = np.mean(masks_val) if masks_val else None
    
    # Build confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy:  {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    if avg_loss is not None:
        print(f"Average Loss: {avg_loss:.4f}")
    if avg_mask is not None:
        print(f"Average Mask Value: {avg_mask:.4f}")
    print(f"\nConfusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Print classification report if class names provided
    if class_names:
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        ))
    
    print("="*70 + "\n")
    
    # Build metrics dictionary
    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm,
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN)
    }
    
    if avg_loss is not None:
        metrics['loss'] = float(avg_loss)
    
    if avg_mask is not None:
        metrics['avg_mask_value'] = float(avg_mask)
    
    return metrics


def eval_val_loss(
    model: FAMIC,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
    use_mask: bool = False,
    use_shift1: bool = False,
    use_shift2: bool = False
) -> Tuple[float, float]:
    """
    Evaluate model and return validation loss and accuracy.
    
    This function matches the original eval_val_loss signature for compatibility.
    
    Args:
        model: Trained FAMIC model
        data_loader: DataLoader for the dataset
        device: Device to run evaluation on (default: cuda if available, else cpu)
        criterion: Loss function (default: nn.BCEWithLogitsLoss())
        use_mask: Whether to use learned mask in forward pass
        use_shift1: Whether to use first shifter in forward pass
        use_shift2: Whether to use second shifter in forward pass
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    metrics = evaluate_model(
        model=model,
        dataloader=data_loader,
        device=device,
        criterion=criterion,
        use_mask=use_mask,
        use_shift1=use_shift1,
        use_shift2=use_shift2,
        return_loss=True
    )
    
    return metrics.get('loss', 0.0), metrics['accuracy']


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
    batch_size: int = 100,
    max_len: int = 150,
    text_column: str = "preprocessed_text",
    label_column: str = "labels",
    use_mask: bool = False,
    use_shift1: bool = False,
    use_shift2: bool = False,
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple datasets using test splits.
    
    Args:
        model: Trained FAMIC model
        dataset_names: List of dataset names to evaluate on
        data_root: Root directory containing datasets
        device: Device to run evaluation on
        batch_size: Batch size for evaluation (default: 100)
        max_len: Maximum sequence length (default: 150)
        text_column: Name of text column in dataset
        label_column: Name of label column in dataset
        use_mask: Whether to use learned mask in forward pass
        use_shift1: Whether to use first shifter in forward pass
        use_shift2: Whether to use second shifter in forward pass
        save_results: Whether to save results to files
        results_dir: Directory to save results
        
    Returns:
        Dictionary mapping dataset names to their evaluation metrics
    """
    from src.datasets import create_dataloaders
    
    all_results = {}
    
    for dataset_name in dataset_names:
        print(f"\n{'='*70}")
        print(f"Evaluating on {dataset_name} dataset...")
        print("="*70)
        
        # Create test dataloader (we only need test, but create_dataloaders creates all three)
        # We'll use the test_loader from the returned tuple
        _, _, test_loader = create_dataloaders(
            dataset_name=dataset_name,
            text_column=text_column,
            label_column=label_column,
            max_len=max_len,
            batch_size=batch_size,
            test_size=0.1,
            val_size=0.5,
            random_state=2025,
            num_workers=0,
            cache_dir=data_root
        )
        
        # Evaluate
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            use_mask=use_mask,
            use_shift1=use_shift1,
            use_shift2=use_shift2
        )
        
        all_results[dataset_name] = metrics
        
        # Save confusion matrix plot
        if save_results:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            cm_path = Path(results_dir) / f"{dataset_name}_confusion_matrix.png"
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=str(cm_path),
                title=f"Confusion Matrix - {dataset_name}"
            )
    
    # Save summary results
    if save_results:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(results_dir) / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("FAMIC Model Evaluation Summary\n")
            f.write("=" * 70 + "\n\n")
            
            for dataset_name, metrics in all_results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                if 'loss' in metrics:
                    f.write(f"  Loss:      {metrics['loss']:.4f}\n")
                f.write(f"  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}\n")
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

