"""
Dataset Loading Utilities

This module provides utilities for loading the datasets used in the FAMIC paper.
Supports loading datasets from HuggingFace and provides basic dataset statistics.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, login
import os
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def authenticate_huggingface(token: Optional[str] = None) -> None:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HuggingFace token. If None, will try to use HF_TOKEN environment variable
               or prompt for login.
    
    Example:
        >>> authenticate_huggingface(token="hf_...")
        >>> # Or set environment variable: export HF_TOKEN=your_token
    """
    if token:
        login(token=token)
        print("✓ Authenticated with HuggingFace using provided token")
    elif os.getenv("HF_TOKEN"):
        print("✓ Using HF_TOKEN environment variable for authentication")
    else:
        print("Attempting to login to HuggingFace...")
        print("If you have a token, you can also set it as an environment variable:")
        print("  export HF_TOKEN=your_token  # Linux/Mac")
        print("  $env:HF_TOKEN='your_token'  # Windows PowerShell")
        try:
            login()
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            print(f"⚠ Authentication failed: {e}")
            print("You may need to authenticate manually or set HF_TOKEN environment variable")


# Dataset registry - maps dataset names to their HuggingFace paths
DATASET_REGISTRY = {
    "twitter": {
        "repo_id": "ycy198888/jds_support_files",
        "filename": "datasets/twitter_cleaned2024.csv",
        "tokenizer_path": "tokenizers/twitter_famic_tokenizer.json",
        "name": "Twitter Cleaned 2024",
        "description": "Twitter dataset cleaned in 2024"
    },
    "wine": {
        "repo_id": "ycy198888/jds_support_files",
        "filename": "datasets/wine_140k_cleaned2025.csv",
        "tokenizer_path": "tokenizers/wine_famic_tokenizer.json",
        "name": "Wine 140k Cleaned 2025",
        "description": "Wine dataset with 140k samples cleaned in 2025"
    }
}


def get_dataset_path(dataset_name: str, cache_dir: Optional[str] = None) -> Path:
    """
    Download and return the path to a dataset file.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files. Defaults to "data" directory.
        
    Returns:
        Path to the downloaded dataset file
        
    Raises:
        ValueError: If dataset_name is not in the registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("data")
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists locally (with new path structure)
    local_path = cache_dir / dataset_info["filename"]
    if local_path.exists():
        print(f"Using cached dataset: {local_path}")
        return local_path
    
    # Also check old location for backward compatibility (filename without datasets/ prefix)
    filename_only = Path(dataset_info["filename"]).name
    old_local_path = cache_dir / filename_only
    if old_local_path.exists():
        print(f"Using cached dataset (old location): {old_local_path}")
        return old_local_path
    
    # Download from HuggingFace
    print(f"Downloading dataset '{dataset_name}' from HuggingFace...")
    print(f"  Repository: {dataset_info['repo_id']}")
    print(f"  Repository type: dataset")
    print(f"  File path: {dataset_info['filename']}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=dataset_info["repo_id"],
            filename=dataset_info["filename"],
            repo_type="dataset",  # This is a dataset repository, not a model repository
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN")  # Use HF_TOKEN environment variable if set
        )
        print(f"✓ Dataset downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg or "Repository Not Found" in error_msg:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}': Authentication required.\n"
                f"The repository may be private or gated. Please authenticate with HuggingFace:\n\n"
                f"Option 1: Set environment variable (recommended):\n"
                f"  export HF_TOKEN=your_huggingface_token\n"
                f"  (On Windows PowerShell: $env:HF_TOKEN='your_huggingface_token')\n\n"
                f"Option 2: Use huggingface_hub login:\n"
                f"  from huggingface_hub import login\n"
                f"  login(token='your_huggingface_token')\n\n"
                f"Option 3: Use CLI:\n"
                f"  huggingface-cli login\n\n"
                f"Get your token from: https://huggingface.co/settings/tokens\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}': {error_msg}\n"
                f"Please check your internet connection and HuggingFace access."
            ) from e


def get_tokenizer_path(dataset_name: str, cache_dir: Optional[str] = None) -> Path:
    """
    Download and return the path to a tokenizer file.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files. Defaults to "data" directory.
        
    Returns:
        Path to the downloaded tokenizer file
        
    Raises:
        ValueError: If dataset_name is not in the registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("data")
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tokenizers subdirectory
    tokenizers_dir = cache_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from tokenizer_path
    tokenizer_filename = Path(dataset_info["tokenizer_path"]).name
    
    # Check if file already exists locally
    local_path = tokenizers_dir / tokenizer_filename
    if local_path.exists():
        print(f"Using cached tokenizer: {local_path}")
        return local_path
    
    # Download from HuggingFace
    print(f"Downloading tokenizer for '{dataset_name}' from HuggingFace...")
    print(f"  Repository: {dataset_info['repo_id']}")
    print(f"  Repository type: dataset")
    print(f"  File path: {dataset_info['tokenizer_path']}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=dataset_info["repo_id"],
            filename=dataset_info["tokenizer_path"],
            repo_type="dataset",  # This is a dataset repository, not a model repository
            cache_dir=str(cache_dir),
            local_dir=str(tokenizers_dir),
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN")  # Use HF_TOKEN environment variable if set
        )
        print(f"✓ Tokenizer downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg or "Repository Not Found" in error_msg:
            raise RuntimeError(
                f"Failed to download tokenizer for '{dataset_name}': Authentication required.\n"
                f"The repository may be private or gated. Please authenticate with HuggingFace:\n\n"
                f"Option 1: Set environment variable (recommended):\n"
                f"  export HF_TOKEN=your_huggingface_token\n"
                f"  (On Windows PowerShell: $env:HF_TOKEN='your_huggingface_token')\n\n"
                f"Option 2: Use huggingface_hub login:\n"
                f"  from huggingface_hub import login\n"
                f"  login(token='your_huggingface_token')\n\n"
                f"Option 3: Use CLI:\n"
                f"  huggingface-cli login\n\n"
                f"Get your token from: https://huggingface.co/settings/tokens\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to download tokenizer for '{dataset_name}': {error_msg}\n"
                f"Please check your internet connection and HuggingFace access."
            ) from e


def load_tokenizer(dataset_name: str, cache_dir: Optional[str] = None):
    """
    Load a Keras tokenizer for a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Keras tokenizer object
        
    Example:
        >>> tokenizer = load_tokenizer("twitter")
        >>> sequences = tokenizer.texts_to_sequences(["Hello world"])
    """
    tokenizer_path = get_tokenizer_path(dataset_name, cache_dir)
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
        
        tokenizer = tokenizer_from_json(tokenizer_json)
        print(f"✓ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{dataset_name}': {e}\n"
            f"Tokenizer file: {tokenizer_path}"
        ) from e


def clear_dataset_cache(dataset_name: str, cache_dir: Optional[str] = None) -> bool:
    """
    Remove cached dataset file to force re-download.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory where cached files are stored. Defaults to "data" directory.
        
    Returns:
        True if cache was cleared, False if no cache found
        
    Example:
        >>> clear_dataset_cache("twitter")
        >>> # Next time you load, it will download fresh data
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("data")
    else:
        cache_dir = Path(cache_dir)
    
    # Check both new and old cache locations
    local_path = cache_dir / dataset_info["filename"]
    filename_only = Path(dataset_info["filename"]).name
    old_local_path = cache_dir / filename_only
    
    cache_cleared = False
    
    # Remove from new location
    if local_path.exists():
        local_path.unlink()
        print(f"✓ Removed cached dataset: {local_path}")
        cache_cleared = True
    
    # Remove from old location (backward compatibility)
    if old_local_path.exists():
        old_local_path.unlink()
        print(f"✓ Removed cached dataset (old location): {old_local_path}")
        cache_cleared = True
    
    if not cache_cleared:
        print(f"No cached dataset found for '{dataset_name}'")
        return False
    
    return True


def clear_tokenizer_cache(dataset_name: str, cache_dir: Optional[str] = None) -> bool:
    """
    Remove cached tokenizer file to force re-download.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory where cached files are stored. Defaults to "data" directory.
        
    Returns:
        True if cache was cleared, False if no cache found
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("data")
    else:
        cache_dir = Path(cache_dir)
    
    # Tokenizers are stored in data/tokenizers/
    tokenizers_dir = cache_dir / "tokenizers"
    tokenizer_filename = Path(dataset_info["tokenizer_path"]).name
    tokenizer_path = tokenizers_dir / tokenizer_filename
    
    if tokenizer_path.exists():
        tokenizer_path.unlink()
        print(f"✓ Removed cached tokenizer: {tokenizer_path}")
        return True
    else:
        print(f"No cached tokenizer found for '{dataset_name}'")
        return False


def clear_all_cache(dataset_name: str, cache_dir: Optional[str] = None, clear_hf_cache: bool = False) -> Dict[str, bool]:
    """
    Remove all cached files (dataset and tokenizer) for a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory where cached files are stored. Defaults to "data" directory.
        clear_hf_cache: If True, also clears HuggingFace's internal cache (default: False)
        
    Returns:
        Dictionary with 'dataset' and 'tokenizer' keys indicating what was cleared
    """
    result = {
        'dataset': clear_dataset_cache(dataset_name, cache_dir),
        'tokenizer': clear_tokenizer_cache(dataset_name, cache_dir)
    }
    
    # Optionally clear HuggingFace's internal cache
    if clear_hf_cache:
        try:
            from huggingface_hub import scan_cache_dir, delete_revisions
            cache_info = scan_cache_dir()
            
            # Find and delete cache for this repo
            repo_id = DATASET_REGISTRY[dataset_name]["repo_id"]
            revisions_to_delete = [
                rev for repo in cache_info.repos 
                for rev in repo.revisions 
                if str(repo.repo_id) == repo_id
            ]
            
            if revisions_to_delete:
                delete_revisions(revisions_to_delete)
                print(f"✓ Cleared HuggingFace internal cache for {repo_id}")
                result['hf_cache'] = True
            else:
                result['hf_cache'] = False
        except Exception as e:
            print(f"⚠ Could not clear HuggingFace cache: {e}")
            result['hf_cache'] = False
    
    if result['dataset'] or result['tokenizer']:
        print(f"\n✓ Cache cleared for '{dataset_name}'. Next load will download fresh data.")
    else:
        print(f"\nNo cache found for '{dataset_name}'")
    
    return result


def load_dataset_csv(dataset_name: str, cache_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load a dataset as a pandas DataFrame.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files
        **kwargs: Additional arguments passed to pd.read_csv()
        
    Returns:
        pandas DataFrame containing the dataset
    """
    dataset_path = get_dataset_path(dataset_name, cache_dir)
    print(f"Loading dataset from: {dataset_path}")
    
    df = pd.read_csv(dataset_path, **kwargs)
    print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def report_dataset_statistics(dataset_name: str, df: Optional[pd.DataFrame] = None, 
                             cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Report basic statistics about a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        df: Optional DataFrame. If provided, uses this instead of loading from file.
        cache_dir: Directory to cache downloaded files (only used if df is None)
        
    Returns:
        Dictionary containing dataset statistics
    """
    # Load dataset if not provided
    if df is None:
        df = load_dataset_csv(dataset_name, cache_dir)
    
    # Get dataset metadata
    dataset_info = DATASET_REGISTRY[dataset_name]
    
    # Calculate statistics
    stats = {
        "dataset_name": dataset_name,
        "display_name": dataset_info["name"],
        "description": dataset_info["description"],
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "shape": df.shape
    }
    
    return stats


def print_dataset_statistics(dataset_name: str, df: Optional[pd.DataFrame] = None,
                            cache_dir: Optional[str] = None) -> None:
    """
    Print formatted dataset statistics to console.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        df: Optional DataFrame. If provided, uses this instead of loading from file.
        cache_dir: Directory to cache downloaded files (only used if df is None)
    """
    stats = report_dataset_statistics(dataset_name, df, cache_dir)
    
    print("\n" + "="*70)
    print(f"Dataset Statistics: {stats['display_name']}")
    print("="*70)
    print(f"Description: {stats['description']}")
    print(f"Total Rows: {stats['total_rows']:,}")
    print(f"Total Columns: {stats['total_columns']}")
    print(f"Shape: {stats['shape'][0]:,} rows × {stats['shape'][1]} columns")
    print(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"Duplicate Rows: {stats['duplicate_rows']:,}")
    print("\nColumn Names:")
    for i, col in enumerate(stats['column_names'], 1):
        dtype = stats['column_types'][col]
        missing = stats['missing_values'][col]
        missing_pct = stats['missing_percentage'][col]
        print(f"  {i:2d}. {col:30s} ({dtype:15s}) - Missing: {missing:,} ({missing_pct:.2f}%)")
    
    if stats['duplicate_rows'] > 0:
        print(f"\n⚠ Warning: {stats['duplicate_rows']:,} duplicate rows found")
    
    print("="*70 + "\n")


class FAMICDataset(Dataset):
    """
    Base dataset class for FAMIC model evaluation.
    
    TODO: Replace with actual dataset implementation.
    This is a placeholder that should be adapted to your specific datasets.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
        label_column: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to the dataset CSV file or directory containing CSV
            split: Dataset split ("train", "val", "test") - currently not used, loads full dataset
            transform: Optional data transformation/augmentation
            label_column: Name of the label column. If None, uses the last column.
            **kwargs: Additional dataset-specific parameters
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.label_column = label_column
        
        # Load actual data from CSV
        self._load_data()
    
    def _load_data(self):
        """
        Load data from CSV files.
        
        Assumes CSV format where:
        - All columns except the last are features
        - Last column is the label/target
        - If label_column is specified in kwargs, uses that column as label
        """
        # Check if data_path is a file or directory
        if self.data_path.is_file():
            csv_path = self.data_path
        elif self.data_path.is_dir():
            # Try to find CSV file in directory
            csv_files = list(self.data_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            csv_path = csv_files[0]  # Use first CSV file found
        else:
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Determine label column (default: last column)
        label_column = self.__dict__.get('label_column', None)
        if label_column is None:
            # Use last column as label by default
            label_column = df.columns[-1]
        
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != label_column]
        self.data = df[feature_columns].values.astype(np.float32)
        self.labels = df[label_column].values
        
        # Convert labels to numeric if they're strings
        if self.labels.dtype == 'object':
            unique_labels = pd.unique(self.labels)
            label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.labels = np.array([label_map[label] for label in self.labels])
        
        self.labels = self.labels.astype(np.int64)
        
        print(f"Loaded {len(self.data)} samples with {self.data.shape[1]} features")
        print(f"Label column: {label_column}, Unique labels: {len(np.unique(self.labels))}")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        features = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        # Apply transforms if provided
        if self.transform:
            features = self.transform(features)
        
        return features, label


def texts_to_padded_word_ids(texts, tokenizer, max_len):
    """
    Convert texts to padded word IDs using tokenizer.
    
    Args:
        texts: List of text strings
        tokenizer: Keras tokenizer
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (padded_sequences, attention_mask)
        - padded_sequences: numpy array of shape (N, max_len) with word IDs
        - attention_mask: numpy array of shape (N, max_len) with 1 for real tokens, 0 for padding
    """
    seqs = tokenizer.texts_to_sequences([str(t) for t in texts])
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post", value=0)
    attn = (padded != 0).astype(np.int64)
    return padded.astype(np.int64), attn


class WordDataset(Dataset):
    """
    PyTorch Dataset for text classification with word-level tokenization.
    
    This dataset class handles:
    - Text tokenization using Keras tokenizer
    - Sequence padding and truncation
    - Attention mask generation
    - Label conversion
    """
    
    def __init__(self, X_text, y, tokenizer, max_len=256):
        """
        Initialize WordDataset.
        
        Args:
            X_text: Array or list of text strings
            y: Array or list of labels
            tokenizer: Keras tokenizer for text tokenization
            max_len: Maximum sequence length (default: 256)
        """
        self.X_text = X_text
        self.y = y
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # Convert texts to padded word IDs
        self.word_ids, self.attn = texts_to_padded_word_ids(self.X_text, self.tokenizer, self.max_len)
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.X_text)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - word_ids: torch.Tensor of shape [L] with token IDs
            - attention_mask: torch.Tensor of shape [L] with 1 for real tokens, 0 for padding
            - labels: torch.Tensor scalar float with label
        """
        return {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),          # [L]
            "attention_mask": torch.tensor(self.attn[idx], dtype=torch.long),        # [L]
            "labels": torch.tensor(float(self.y[idx]), dtype=torch.float32),         # scalar float
        }


def create_train_val_test_split(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = 0.1,
    val_size: float = 0.5,
    random_state: int = 2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test splits from a DataFrame.
    
    Args:
        df: DataFrame containing the data
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        test_size: Proportion of data to use for test set (default: 0.1)
        val_size: Proportion of test set to use for validation (default: 0.5)
                 Note: This means val_size of the test_size, so final split is:
                 train: 1 - test_size
                 val: test_size * val_size
                 test: test_size * (1 - val_size)
        random_state: Random seed for reproducibility (default: 2025)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Extract data
    X_data = np.array(df[text_column])
    y_data = np.array(df[label_column])
    
    # First split: train vs (val + test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    
    # Second split: val vs test (from the test portion)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=val_size, random_state=random_state
    )
    
    print("Data Split done.")
    print(f"train/val/test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"train label mean: {y_train.mean():.4f}, val: {y_val.mean():.4f}, test: {y_test.mean():.4f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(
    dataset_name: str,
    text_column: str = "preprocessed_text",
    label_column: str = "labels",
    max_len: int = 150,
    batch_size: int = 100,
    test_size: float = 0.1,
    val_size: float = 0.5,
    random_state: int = 2025,
    num_workers: int = 0,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test dataloaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        text_column: Name of the column containing text data (default: "preprocessed_text")
        label_column: Name of the column containing labels (default: "labels")
        max_len: Maximum sequence length (default: 150)
        batch_size: Batch size for DataLoaders (default: 100)
        test_size: Proportion of data for test set (default: 0.1)
        val_size: Proportion of test set for validation (default: 0.5)
        random_state: Random seed for reproducibility (default: 2025)
        num_workers: Number of worker processes for DataLoader (default: 0)
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    df = load_dataset_csv(dataset_name, cache_dir=cache_dir)
    
    # Check if required columns exist
    if text_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Text column '{text_column}' not found in dataset. "
            f"Available columns: {available}"
        )
    
    if label_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Label column '{label_column}' not found in dataset. "
            f"Available columns: {available}"
        )
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        df=df,
        text_column=text_column,
        label_column=label_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer(dataset_name, cache_dir=cache_dir)
    
    # Create datasets
    train_ds = WordDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_ds = WordDataset(X_val, y_val, tokenizer, max_len=max_len)
    test_ds = WordDataset(X_test, y_test, tokenizer, max_len=max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def load_dataset(
    dataset_name: str,
    data_root: str = "data",
    split: str = "test",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Load a dataset and return a DataLoader.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        data_root: Root directory containing datasets
        split: Dataset split ("train", "val", "test") - currently all data is loaded
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments passed to dataset
        
    Returns:
        DataLoader for the specified dataset and split
    """
    # Validate dataset name
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    # Get dataset path (downloads if needed)
    dataset_path = get_dataset_path(dataset_name, cache_dir=data_root)
    dataset = FAMICDataset(dataset_path, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def get_dataset_info(dataset_name: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Dictionary containing dataset information
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    # Load dataset to get actual statistics
    df = load_dataset_csv(dataset_name, cache_dir)
    stats = report_dataset_statistics(dataset_name, df, cache_dir)
    
    # Return formatted info
    info = {
        "name": dataset_name,
        "display_name": stats["display_name"],
        "description": stats["description"],
        "total_rows": stats["total_rows"],
        "total_columns": stats["total_columns"],
        "column_names": stats["column_names"],
        "shape": stats["shape"],
        "memory_usage_mb": stats["memory_usage_mb"],
        # These may need to be updated based on actual dataset structure
        "num_classes": 2,  # TODO: Update based on actual dataset
        "input_dim": stats["total_columns"] - 1,  # TODO: Update based on actual structure
        "num_samples": {
            "train": 0,  # TODO: Update if train/val/test splits are available
            "val": 0,
            "test": stats["total_rows"]  # Using total rows as test for now
        }
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Dataset Loading Utilities")
    print("="*70)
    
    # Example 1: Load and report statistics for Twitter dataset
    print("\nExample 1: Loading Twitter dataset...")
    try:
        df_twitter = load_dataset_csv("twitter")
        print_dataset_statistics("twitter", df=df_twitter)
    except Exception as e:
        print(f"Error loading Twitter dataset: {e}")
    
    # Example 2: Load and report statistics for Wine dataset
    print("\nExample 2: Loading Wine dataset...")
    try:
        df_wine = load_dataset_csv("wine")
        print_dataset_statistics("wine", df=df_wine)
    except Exception as e:
        print(f"Error loading Wine dataset: {e}")
    
    # Example 3: Get dataset info
    print("\nExample 3: Getting dataset info...")
    try:
        info = get_dataset_info("twitter")
        print(f"Dataset: {info['display_name']}")
        print(f"Total rows: {info['total_rows']:,}")
        print(f"Columns: {info['total_columns']}")
    except Exception as e:
        print(f"Error getting dataset info: {e}")
    
    # Example 4: Load tokenizers
    print("\nExample 4: Loading tokenizers...")
    for dataset_name in ["twitter", "wine"]:
        try:
            print(f"\nLoading tokenizer for '{dataset_name}' dataset...")
            tokenizer = load_tokenizer(dataset_name)
            vocab_size = len(tokenizer.word_index) if hasattr(tokenizer, 'word_index') else 'N/A'
            print(f"✓ Tokenizer loaded successfully (vocabulary size: {vocab_size})")
        except Exception as e:
            print(f"Error loading tokenizer for {dataset_name}: {e}")

