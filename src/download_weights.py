"""
Model Weights Download Utility

This module handles automatic downloading of FAMIC model weights.
TODO: Update with actual download URL and authentication if needed.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import hashlib


def verify_checksum(file_path: str, expected_checksum: Optional[str] = None) -> bool:
    """
    Verify file integrity using SHA256 checksum.
    
    Args:
        file_path: Path to the file to verify
        expected_checksum: Expected SHA256 checksum (optional)
        
    Returns:
        True if checksum matches or if no expected checksum provided
    """
    if expected_checksum is None:
        return True
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_checksum = sha256_hash.hexdigest()
    return actual_checksum.lower() == expected_checksum.lower()


def download_weights(
    url: str,
    save_path: str,
    expected_checksum: Optional[str] = None,
    force_download: bool = False
) -> str:
    """
    Download model weights from a URL.
    
    Args:
        url: URL to download weights from
            TODO: Replace with actual model weights URL
            Example: "https://example.com/models/famic_weights.pth"
        save_path: Local path to save the weights file
        expected_checksum: Optional SHA256 checksum to verify file integrity
        force_download: If True, re-download even if file exists
        
    Returns:
        Path to the downloaded weights file
        
    Raises:
        urllib.error.URLError: If download fails
        ValueError: If checksum verification fails
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if save_path.exists() and not force_download:
        print(f"Weights file already exists at {save_path}")
        if expected_checksum:
            if verify_checksum(str(save_path), expected_checksum):
                print("Checksum verification passed.")
                return str(save_path)
            else:
                print("Checksum verification failed. Re-downloading...")
        else:
            return str(save_path)
    
    # Download the file
    print(f"Downloading weights from {url}...")
    try:
        urllib.request.urlretrieve(url, str(save_path))
        print(f"Downloaded weights to {save_path}")
    except urllib.error.URLError as e:
        raise urllib.error.URLError(
            f"Failed to download weights from {url}: {e}"
        )
    
    # Verify checksum if provided
    if expected_checksum:
        if verify_checksum(str(save_path), expected_checksum):
            print("Checksum verification passed.")
        else:
            raise ValueError(
                f"Checksum verification failed. "
                f"Expected: {expected_checksum}, "
                f"Got: {hashlib.sha256(open(save_path, 'rb').read()).hexdigest()}"
            )
    
    return str(save_path)


def get_weights_path(
    weights_dir: str = "models",
    weights_filename: str = "famic_weights.pth"
) -> str:
    """
    Get the path to model weights, downloading if necessary.
    
    Args:
        weights_dir: Directory to store weights (default: "models")
        weights_filename: Name of the weights file (default: "famic_weights.pth")
        
    Returns:
        Path to the weights file
    """
    weights_path = Path(weights_dir) / weights_filename
    
    # TODO: Replace with actual download URL
    # TODO: Add expected_checksum if available
    download_url = "https://example.com/models/famic_weights.pth"  # PLACEHOLDER
    expected_checksum = None  # TODO: Add SHA256 checksum when available
    
    if not weights_path.exists():
        print(f"Weights not found at {weights_path}")
        print("Please update download_weights.py with the actual download URL")
        print(f"Expected URL: {download_url}")
        raise FileNotFoundError(
            f"Model weights not found. Please download from: {download_url}\n"
            f"Or update the download URL in src/download_weights.py"
        )
        # Uncomment when URL is available:
        # download_weights(download_url, str(weights_path), expected_checksum)
    
    return str(weights_path)


if __name__ == "__main__":
    # Example usage
    weights_path = get_weights_path()
    print(f"Model weights available at: {weights_path}")

