"""
FAMIC Model Evaluation Package

This package provides utilities for:
- Initializing and loading the FAMIC model
- Downloading model weights from HuggingFace
- Loading datasets and tokenizers from HuggingFace
- Creating train/validation/test splits
- Evaluating model performance with comprehensive metrics
"""

__version__ = "0.1.0"

# Model exports
from src.model import (
    FAMIC,
    create_embedding_matrix,
    create_emb_layer,
    initialize_model_blocks,
    EMBEDDING_DIMENSIONS,
    VOCAB_LENGTH,
    MAX_LEN,
    NUM_HEADS,
)

# Dataset exports
from src.datasets import (
    DATASET_REGISTRY,
    load_dataset_csv,
    load_tokenizer,
    get_dataset_path,
    get_tokenizer_path,
    get_dataset_info,
    report_dataset_statistics,
    print_dataset_statistics,
    create_train_val_test_split,
    create_dataloaders,
    WordDataset,
    clear_dataset_cache,
    clear_tokenizer_cache,
    clear_all_cache,
    authenticate_huggingface,
)

# Evaluation exports
from src.evaluate import (
    evaluate_model,
    eval_val_loss,
    plot_confusion_matrix,
    evaluate_on_datasets,
)

# Weight loading exports
from src.download_weights import (
    WEIGHTS_REGISTRY,
    get_weights_path,
    download_all_weights,
    load_pretrained_weights,
)

__all__ = [
    # Model
    "FAMIC",
    "create_embedding_matrix",
    "create_emb_layer",
    "initialize_model_blocks",
    "EMBEDDING_DIMENSIONS",
    "VOCAB_LENGTH",
    "MAX_LEN",
    "NUM_HEADS",
    # Datasets
    "DATASET_REGISTRY",
    "load_dataset_csv",
    "load_tokenizer",
    "get_dataset_path",
    "get_tokenizer_path",
    "get_dataset_info",
    "report_dataset_statistics",
    "print_dataset_statistics",
    "create_train_val_test_split",
    "create_dataloaders",
    "WordDataset",
    "clear_dataset_cache",
    "clear_tokenizer_cache",
    "clear_all_cache",
    "authenticate_huggingface",
    # Evaluation
    "evaluate_model",
    "eval_val_loss",
    "plot_confusion_matrix",
    "evaluate_on_datasets",
    # Weights
    "WEIGHTS_REGISTRY",
    "get_weights_path",
    "download_all_weights",
    "load_pretrained_weights",
]

