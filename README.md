# FAMIC Model Evaluation

This repository provides tools for evaluating the FAMIC model as described in our Journal of Data Science submission.

## Features

- ✅ Initialize the FAMIC model through PyTorch
- ✅ Automatically download model weights (URL to be configured)
- ✅ Load and evaluate on two datasets discussed in the paper
- ✅ Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd FAMIC
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Model Weights

Before running evaluation, you need to set up the model weights:

1. **Option A: Automatic Download** (when URL is available)
   - Update `src/download_weights.py` with the actual download URL
   - The script will automatically download weights on first run

2. **Option B: Manual Download**
   - Download model weights from the provided source
   - Place the weights file in `models/famic_weights.pth` (or update path in config)

### 2. Prepare Datasets

Place your datasets in the following structure:
```
data/
├── dataset1/
│   ├── test/  # Test split
│   ├── train/ # Optional: training split
│   └── val/   # Optional: validation split
└── dataset2/
    ├── test/
    ├── train/
    └── val/
```

**TODO**: Update `src/datasets.py` with your actual dataset loading logic.

### 3. Update Configuration

Edit `config/config.yaml` with your specific:
- Model parameters (input_dim, hidden_dim, num_classes, etc.)
- Dataset paths and information
- Evaluation settings

### 4. Run Evaluation

```bash
python scripts/evaluate.py
```

Or use the evaluation module directly:
```python
from src.model import FAMIC
from src.download_weights import get_weights_path
from src.evaluate import evaluate_on_datasets
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = get_weights_path()
model = FAMIC.from_pretrained(weights_path, device=device)

results = evaluate_on_datasets(
    model,
    dataset_names=["dataset1", "dataset2"],
    device=device
)
```

Or see `examples/example_usage.py` for a step-by-step example.

## Project Structure

```
FAMIC/
├── src/
│   ├── __init__.py
│   ├── model.py              # FAMIC model definition
│   ├── download_weights.py   # Model weights download utility
│   ├── datasets.py           # Dataset loading utilities
│   └── evaluate.py           # Evaluation metrics and utilities
├── config/
│   └── config.yaml           # Configuration file
├── scripts/
│   └── evaluate.py           # Main evaluation script
├── examples/
│   └── example_usage.py      # Example usage script
├── tests/                    # Unit tests
├── data/                     # Dataset directory (gitignored)
├── models/                   # Model weights directory (gitignored)
├── results/                  # Evaluation results (gitignored)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

### Model Configuration

Update `config/config.yaml` or modify model parameters directly:

```yaml
model:
  input_dim: 512      # TODO: Update with actual input dimension
  hidden_dim: 256     # TODO: Update with actual hidden dimension
  num_classes: 2      # TODO: Update with actual number of classes
```

### Dataset Configuration

Configure your datasets in `config/config.yaml`:

```yaml
datasets:
  data_root: "data"
  dataset1:
    name: "dataset1"
    path: "data/dataset1"
    num_classes: 2
    input_dim: 512
```

## Evaluation Metrics

The evaluation script reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1-score
- **Confusion Matrix**: Visualized and saved as PNG

Results are saved in the `results/` directory:
- `evaluation_summary.txt`: Text summary of all metrics
- `{dataset_name}_confusion_matrix.png`: Confusion matrix visualization

## TODO Checklist

Before using this repository, please complete the following:

- [ ] **Model Architecture**: Replace placeholder in `src/model.py` with actual FAMIC architecture
- [ ] **Model Weights**: Update download URL in `src/download_weights.py` or provide manual download instructions
- [ ] **Dataset Loading**: Implement actual dataset loading logic in `src/datasets.py` for both datasets
- [ ] **Configuration**: Update `config/config.yaml` with actual model and dataset parameters
- [ ] **Class Names**: Add class names to config if you want labeled confusion matrices
- [ ] **Documentation**: Add dataset-specific documentation and citation information

## Development

### Adding a New Dataset

1. Update `src/datasets.py`:
   - Add dataset-specific loading logic in `load_dataset()`
   - Update `get_dataset_info()` with dataset metadata

2. Update `config/config.yaml`:
   - Add dataset configuration entry

### Customizing Evaluation

Modify `src/evaluate.py` to:
- Add additional metrics
- Change visualization style
- Add per-class metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{famic2024,
  title={FAMIC: [Your Paper Title]},
  author={[Authors]},
  journal={Journal of Data Science},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

[Add contact information or issue tracker link]

