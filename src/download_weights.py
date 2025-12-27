"""
Model Weights Download Utility

This module handles automatic downloading of FAMIC model weights from HuggingFace.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import torch
from huggingface_hub import hf_hub_download


# Model weights registry - maps dataset names to their weight file paths
WEIGHTS_REGISTRY = {
    "twitter": {
        "repo_id": "ycy198888/jds_support_files",
        "base_path": "FAMIC/twitter_pretrained_weights",
        "version": "v2",
        "weights": {
            "embeds": "embedding_weights.pt",
            "sentiment": "three_block_mb1_{version}.pt",
            "mask": "three_block_mb2_{version}.pt",
            "shifter1": "three_block_mb31_{version}.pt",
            "shifter2": "three_block_mb32_{version}.pt",
            "synthesizer": "three_block_mb4_{version}.pt"
        }
    },
    "wine": {
        "repo_id": "ycy198888/jds_support_files",
        "base_path": "FAMIC/wine_pretrained_weights",
        "version": "v2",
        "weights": {
            "embeds": "embedding_weights.pt",
            "sentiment": "three_block_mb1_{version}.pt",
            "mask": "three_block_mb2_{version}.pt",
            "shifter1": "three_block_mb31_{version}.pt",
            "shifter2": "three_block_mb32_{version}.pt",
            "synthesizer": "three_block_mb4_{version}.pt"
        }
    }
}


def get_weights_path(
    dataset_name: str,
    weight_name: str,
    cache_dir: Optional[str] = None,
    version: Optional[str] = None
) -> Path:
    """
    Download and return the path to a model weight file.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        weight_name: Name of the weight file ("embeds", "sentiment", "mask", 
                   "shifter1", "shifter2", "synthesizer")
        cache_dir: Directory to cache downloaded files. Defaults to "models" directory.
        version: Model version (defaults to "v2" from registry)
        
    Returns:
        Path to the downloaded weight file
        
    Raises:
        ValueError: If dataset_name or weight_name is not in the registry
    """
    if dataset_name not in WEIGHTS_REGISTRY:
        available = ", ".join(WEIGHTS_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    weights_info = WEIGHTS_REGISTRY[dataset_name]
    
    if weight_name not in weights_info["weights"]:
        available = ", ".join(weights_info["weights"].keys())
        raise ValueError(
            f"Unknown weight name: '{weight_name}'. "
            f"Available weights: {available}"
        )
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path("models")
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = cache_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Get version
    version = version or weights_info["version"]
    
    # Get weight filename (with version substitution)
    weight_template = weights_info["weights"][weight_name]
    weight_filename = weight_template.format(version=version)
    
    # Construct full path in HuggingFace repo
    repo_filename = f"{weights_info['base_path']}/{weight_filename}"
    
    # Check if file already exists locally
    local_path = dataset_dir / weight_filename
    if local_path.exists():
        print(f"Using cached weight: {local_path}")
        return local_path
    
    # Download from HuggingFace
    print(f"Downloading {weight_name} weights for '{dataset_name}' dataset...")
    print(f"  Repository: {weights_info['repo_id']}")
    print(f"  Repository type: dataset")
    print(f"  File path: {repo_filename}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=weights_info["repo_id"],
            filename=repo_filename,
            repo_type="dataset",  # This is a dataset repository
            cache_dir=str(cache_dir),
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN")  # Use HF_TOKEN environment variable if set
        )
        print(f"✓ Weight downloaded to: {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg or "Repository Not Found" in error_msg:
            raise RuntimeError(
                f"Failed to download weight '{weight_name}' for '{dataset_name}': Authentication required.\n"
                f"The repository may be private or gated. Please authenticate with HuggingFace:\n\n"
                f"Option 1: Set environment variable (recommended):\n"
                f"  export HF_TOKEN=your_huggingface_token\n"
                f"  (On Windows PowerShell: $env:HF_TOKEN='your_huggingface_token')\n\n"
                f"Option 2: Use huggingface_hub login:\n"
                f"  from huggingface_hub import login\n"
                f"  login(token='your_huggingface_token')\n\n"
                f"Get your token from: https://huggingface.co/settings/tokens\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to download weight '{weight_name}' for '{dataset_name}': {error_msg}\n"
                f"Please check your internet connection and HuggingFace access."
            ) from e


def download_all_weights(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    version: Optional[str] = None
) -> Dict[str, Path]:
    """
    Download all model weights for a dataset.
    
    Args:
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files
        version: Model version (defaults to "v2" from registry)
        
    Returns:
        Dictionary mapping weight names to their file paths
    """
    weights_info = WEIGHTS_REGISTRY[dataset_name]
    weight_paths = {}
    
    print(f"Downloading all weights for '{dataset_name}' dataset...")
    print("="*70)
    
    for weight_name in weights_info["weights"].keys():
        weight_paths[weight_name] = get_weights_path(
            dataset_name=dataset_name,
            weight_name=weight_name,
            cache_dir=cache_dir,
            version=version
        )
    
    print("="*70)
    print(f"✓ All weights downloaded for '{dataset_name}' dataset")
    
    return weight_paths


def load_pretrained_weights(
    model_blocks: Dict,
    dataset_name: str,
    cache_dir: Optional[str] = None,
    version: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load pretrained weights into model blocks.
    
    Args:
        model_blocks: Dictionary of model blocks (from initialize_model_blocks)
        dataset_name: Name of the dataset ("twitter" or "wine")
        cache_dir: Directory to cache downloaded files
        version: Model version (defaults to "v2" from registry)
        device: Device to load weights on (default: cuda if available, else cpu)
        
    Returns:
        Dictionary of model blocks with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading pretrained weights for '{dataset_name}' dataset...")
    print(f"  Device: {device}")
    print("="*70)
    
    # Download all weights
    weight_paths = download_all_weights(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        version=version
    )
    
    # Load weights into model blocks
    print("\nLoading weights into model blocks...")
    
    # Map weight names to model block names
    weight_to_block = {
        "embeds": "embeds",
        "sentiment": "sentiment",
        "mask": "mask",
        "shifter1": "shifter1",
        "shifter2": "shifter2",
        "synthesizer": "synthesizer"
    }
    
    for weight_name, block_name in weight_to_block.items():
        if block_name in model_blocks:
            weight_path = weight_paths[weight_name]
            print(f"  Loading {weight_name} -> {block_name}...")
            
            try:
                state_dict = torch.load(weight_path, map_location=device)
                model_blocks[block_name].load_state_dict(state_dict)
                model_blocks[block_name].to(device)
                print(f"    ✓ Loaded from: {weight_path.name}")
            except Exception as e:
                print(f"    ✗ Failed to load {weight_name}: {e}")
                raise RuntimeError(f"Failed to load weight '{weight_name}': {e}") from e
    
    print("="*70)
    print("✓ All pretrained weights loaded successfully")
    
    return model_blocks


def get_weights_path_legacy(
    weights_dir: str = "models",
    weights_filename: str = "famic_weights.pth"
) -> str:
    """
    Legacy function for backward compatibility.
    Get the path to model weights, downloading if necessary.
    
    Args:
        weights_dir: Directory to store weights (default: "models")
        weights_filename: Name of the weights file (default: "famic_weights.pth")
        
    Returns:
        Path to the weights file
    """
    weights_path = Path(weights_dir) / weights_filename
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}.\n"
            f"Please use download_all_weights() or load_pretrained_weights() "
            f"to download weights from HuggingFace."
        )
    
    return str(weights_path)


if __name__ == "__main__":
    # Example usage
    print("Model Weights Download Utility")
    print("="*70)
    
    # Example: Download all weights for Twitter dataset
    try:
        weight_paths = download_all_weights("twitter")
        print("\nDownloaded weights:")
        for name, path in weight_paths.items():
            print(f"  {name}: {path}")
    except Exception as e:
        print(f"Error: {e}")
