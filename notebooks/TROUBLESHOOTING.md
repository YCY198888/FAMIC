# Troubleshooting Guide

## ImportError: cannot import name 'WordDataset'

If you get this error, it means your Jupyter kernel has cached an old version of the `src.datasets` module.

### Solution 1: Restart Kernel (Recommended)
1. Go to `Kernel` → `Restart Kernel` (or press `0` twice in command mode)
2. Re-run all cells from the beginning

### Solution 2: Force Reload Module
Add this code before importing:

```python
import importlib
import sys

# Remove from cache
if 'src.datasets' in sys.modules:
    del sys.modules['src.datasets']

# Re-import
from src.datasets import WordDataset, create_train_val_test_split, create_dataloaders
```

### Solution 3: Use importlib.reload
```python
import importlib
import src.datasets
importlib.reload(src.datasets)

from src.datasets import WordDataset
```

## Why This Happens

When you modify Python files that are already imported, Jupyter doesn't automatically reload them. The kernel keeps the old version in memory. This is a common issue during development.

## Best Practice

When developing and modifying source files:
1. Restart the kernel after making changes to Python modules
2. Or use `importlib.reload()` to force a reload
3. Or use `autoreload` magic command: `%load_ext autoreload` and `%autoreload 2`

