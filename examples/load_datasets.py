"""
Example script demonstrating dataset loading and statistics reporting.

This script shows how to:
1. Load datasets from HuggingFace
2. Switch between different datasets easily
3. Report basic dataset statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import (
    load_dataset_csv,
    print_dataset_statistics,
    report_dataset_statistics,
    get_dataset_info,
    load_tokenizer,
    DATASET_REGISTRY
)


def main():
    """Demonstrate dataset loading and statistics."""
    
    print("="*70)
    print("FAMIC Dataset Loading Example")
    print("="*70)
    
    # Show available datasets
    print("\nAvailable datasets:")
    for name, info in DATASET_REGISTRY.items():
        print(f"  - {name}: {info['name']}")
    
    # Example 1: Load Twitter dataset and show statistics
    print("\n" + "="*70)
    print("Example 1: Loading Twitter Dataset")
    print("="*70)
    try:
        df_twitter = load_dataset_csv("twitter")
        print_dataset_statistics("twitter", df=df_twitter)
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires internet connection to download from HuggingFace")
    
    # Example 2: Load Wine dataset and show statistics
    print("\n" + "="*70)
    print("Example 2: Loading Wine Dataset")
    print("="*70)
    try:
        df_wine = load_dataset_csv("wine")
        print_dataset_statistics("wine", df=df_wine)
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires internet connection to download from HuggingFace")
    
    # Example 3: Get dataset info programmatically
    print("\n" + "="*70)
    print("Example 3: Getting Dataset Info Programmatically")
    print("="*70)
    for dataset_name in DATASET_REGISTRY.keys():
        try:
            info = get_dataset_info(dataset_name)
            print(f"\n{info['display_name']}:")
            print(f"  Total rows: {info['total_rows']:,}")
            print(f"  Total columns: {info['total_columns']}")
            print(f"  Column names: {', '.join(info['column_names'][:5])}..." 
                  if len(info['column_names']) > 5 
                  else f"  Column names: {', '.join(info['column_names'])}")
        except Exception as e:
            print(f"Error getting info for {dataset_name}: {e}")
    
    # Example 4: Load tokenizers
    print("\n" + "="*70)
    print("Example 4: Loading Tokenizers")
    print("="*70)
    for dataset_name in DATASET_REGISTRY.keys():
        try:
            print(f"\nLoading tokenizer for '{dataset_name}' dataset...")
            tokenizer = load_tokenizer(dataset_name)
            print(f"✓ Tokenizer loaded successfully")
            print(f"  Vocabulary size: {len(tokenizer.word_index) if hasattr(tokenizer, 'word_index') else 'N/A'}")
        except Exception as e:
            print(f"Error loading tokenizer for {dataset_name}: {e}")
            print("Note: This requires internet connection to download from HuggingFace")
    
    # Example 5: Easy dataset switching
    print("\n" + "="*70)
    print("Example 5: Easy Dataset Switching")
    print("="*70)
    print("You can easily switch between datasets by changing the dataset name:")
    print("  df = load_dataset_csv('twitter')  # Load Twitter dataset")
    print("  df = load_dataset_csv('wine')     # Load Wine dataset")
    print("  tokenizer = load_tokenizer('twitter')  # Load Twitter tokenizer")
    print("  tokenizer = load_tokenizer('wine')     # Load Wine tokenizer")
    print("\nBoth datasets and tokenizers are automatically downloaded and cached in the 'data' directory.")


if __name__ == "__main__":
    main()

