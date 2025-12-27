# FAMIC Model Evaluation

This repository provides tools for evaluating the FAMIC model as described in our Journal of Data Science submission.

## Features

- ✅ **Complete FAMIC Model Architecture**: Full PyTorch implementation with all components (Embeds, Sentiment, Mask, Shifter blocks, Synthesizer)
- ✅ **Automatic Dataset Loading**: Download and cache datasets from HuggingFace (Twitter and Wine datasets)
- ✅ **Tokenizer Support**: Load Keras tokenizers for text preprocessing
- ✅ **Pretrained Weights**: Automatic download and loading of pretrained model weights from HuggingFace
- ✅ **Data Splitting**: Reproducible train/validation/test splits with configurable random seed
- ✅ **PyTorch DataLoaders**: Efficient data loading with padding and attention masks
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, and detailed classification reports
- ✅ **Cache Management**: Utilities to clear cached datasets and tokenizers for fresh downloads
- ✅ **Jupyter Tutorial**: Complete notebook demonstrating all functionalities

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YCY198888/FAMIC.git
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

### 1. HuggingFace Authentication

**Note**: Authentication is **not required** for the Twitter and Wine datasets used in this repository. The datasets, tokenizers, and model weights are publicly available on HuggingFace.

If you encounter authentication errors or need to access private repositories, you can authenticate with HuggingFace:

```python
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
```

Or use the CLI:
```bash
huggingface-cli login
```

### 2. Load Dataset and Tokenizer

```python
from src.datasets import load_dataset_csv, load_tokenizer

# Load dataset (automatically downloads and caches)
df = load_dataset_csv("twitter")  # or "wine"

# Load tokenizer
tokenizer = load_tokenizer("twitter")
```

### 3. Initialize Model with Pretrained Weights

```python
from src.model import FAMIC, create_embedding_matrix
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create embedding matrix
embedding_matrix = create_embedding_matrix(vocab_length=250000, embedding_dim=300)

# Load model with pretrained weights
model = FAMIC.from_pretrained_huggingface(
    dataset_name="twitter",
    embedding_matrix=embedding_matrix,
    device=device
)
```

### 4. Create Data Splits and DataLoaders

```python
from src.datasets import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset_name="twitter",
    text_column="preprocessed_text",
    label_column="labels",
    max_len=150,
    batch_size=100,
    random_state=2025
)
```

### 5. Evaluate Model

```python
from src.evaluate import evaluate_model
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()

test_metrics = evaluate_model(
    model=model,
    dataloader=test_loader,
    device=device,
    criterion=criterion,
    use_mask=True,
    use_shift1=True,
    use_shift2=True,
    return_loss=True
)

print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"F1-Score: {test_metrics['f1_score']:.4f}")
```

### 6. Clear Cache (if needed)

If you need to re-download datasets after corrections:

```python
from src.datasets import clear_dataset_cache

clear_dataset_cache("twitter")
```

## Project Structure

```
FAMIC/
├── src/
│   ├── __init__.py
│   ├── model.py              # FAMIC model architecture
│   ├── download_weights.py   # Pretrained weights download utility
│   ├── datasets.py           # Dataset loading and preprocessing
│   └── evaluate.py           # Evaluation metrics and utilities
├── config/
│   └── config.yaml           # Configuration file
├── notebooks/
│   ├── famic_tutorial.ipynb       # Complete tutorial notebook (local)
│   └── famic_tutorial_colab.ipynb # Google Colab version
├── examples/
│   └── example_usage.py      # Example usage script
├── data/                     # Dataset cache directory (gitignored)
├── models/                   # Model weights cache directory (gitignored)
├── results/                  # Evaluation results (gitignored)
├── requirements.txt          # Python dependencies
└── README.md
```

## Available Datasets

The repository supports two datasets:

1. **Twitter** (`twitter`): Twitter dataset cleaned in 2024
   - Repository: `ycy198888/jds_support_files`
   - File: `datasets/twitter_cleaned2024.csv`
   - Tokenizer: `tokenizers/twitter_famic_tokenizer.json`
   - Weights: `FAMIC/twitter_pretrained_weights/`

2. **Wine** (`wine`): Wine dataset with 140k samples cleaned in 2025
   - Repository: `ycy198888/jds_support_files`
   - File: `datasets/wine_140k_cleaned2025.csv`
   - Tokenizer: `tokenizers/wine_famic_tokenizer.json`
   - Weights: `FAMIC/wine_pretrained_weights/`

## Model Architecture

The FAMIC model consists of the following components:

- **Embeds**: Embedding layer with pre-trained word embeddings
- **Sentiment_block**: Sentiment analysis block with self-attention
- **Mask_block**: Learned mask generation block
- **Shifter_block1**: First shifter block for position-aware features
- **Shifter_block2**: Second shifter block with relative position encoding
- **Synthesizer**: Final synthesizer block combining all features

### Model Parameters

- **Embedding Dimension**: 300
- **Vocabulary Size**: 250,001 (250,000 + 1 for padding)
- **Max Sequence Length**: 100 (configurable, default 150 for dataloaders)
- **Number of Attention Heads**: 10
- **Hidden Dimension**: 300

## Configuration

### Model Configuration

The model uses the following default parameters (configurable):

```python
EMBEDDING_DIMENSIONS = 300
VOCAB_LENGTH = 250000
MAX_LEN = 100
NUM_HEADS = 10
```

### Dataset Configuration

Datasets are configured in `config/config.yaml`:

```yaml
datasets:
  default_dataset: "twitter"
  twitter:
    repo_id: "ycy198888/jds_support_files"
    filename: "datasets/twitter_cleaned2024.csv"
    tokenizer_path: "tokenizers/twitter_famic_tokenizer.json"
  wine:
    repo_id: "ycy198888/jds_support_files"
    filename: "datasets/wine_140k_cleaned2025.csv"
    tokenizer_path: "tokenizers/wine_famic_tokenizer.json"
```

## Evaluation Metrics

The evaluation module provides:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score for positive class
- **Recall**: Recall score for positive class
- **F1-Score**: F1-score for positive class
- **Confusion Matrix**: Full confusion matrix with visualization
- **Classification Report**: Per-class metrics and support
- **Average Loss**: Average loss across all batches

Results can be visualized and saved as PNG files.

## Jupyter Notebook Tutorial

Comprehensive tutorial notebooks are available:

- **Local Jupyter**: `notebooks/famic_tutorial.ipynb` - For running locally
- **Google Colab**: `notebooks/famic_tutorial_colab.ipynb` - Ready to run in Google Colab (automatically clones repository and installs dependencies)

Both notebooks demonstrate:

1. Environment setup and imports
2. Loading datasets and tokenizers (no authentication required)
3. Model initialization
4. Loading pretrained weights
5. Creating data splits and DataLoaders
6. Model evaluation
7. Visualization of results

## Cache Management

All downloaded files are cached locally:

- **Datasets**: `data/datasets/`
- **Tokenizers**: `data/tokenizers/`
- **Model Weights**: `models/{dataset_name}/FAMIC/{dataset_name}_pretrained_weights/`

To clear cache and force re-download:

```python
from src.datasets import clear_dataset_cache, clear_all_cache

# Clear only dataset cache
clear_dataset_cache("twitter")

# Clear both dataset and tokenizer cache
clear_all_cache("twitter")
```

## API Reference

### Main Functions

#### Dataset Loading
- `load_dataset_csv(dataset_name, cache_dir=None)`: Load dataset as pandas DataFrame
- `load_tokenizer(dataset_name, cache_dir=None)`: Load Keras tokenizer
- `create_train_val_test_split(df, text_column, label_column, test_size=0.1, val_size=0.5, random_state=2025)`: Create data splits
- `create_dataloaders(...)`: Create PyTorch DataLoaders for train/val/test

#### Model Loading
- `FAMIC.from_pretrained_huggingface(...)`: Load model with pretrained weights
- `load_pretrained_weights(...)`: Load weights into existing model blocks

#### Evaluation
- `evaluate_model(...)`: Comprehensive model evaluation
- `eval_val_loss(...)`: Simple evaluation returning loss and accuracy
- `plot_confusion_matrix(...)`: Visualize confusion matrix

#### Cache Management
- `clear_dataset_cache(dataset_name)`: Clear cached dataset
- `clear_tokenizer_cache(dataset_name)`: Clear cached tokenizer
- `clear_all_cache(dataset_name)`: Clear all caches for a dataset

## Development

### Adding a New Dataset

1. Update `src/datasets.py`:
   - Add entry to `DATASET_REGISTRY` with repository info
   - Ensure tokenizer path is correct

2. Update `config/config.yaml`:
   - Add dataset configuration entry

3. Add pretrained weights to HuggingFace:
   - Upload weights to `FAMIC/{dataset_name}_pretrained_weights/`

### Customizing Evaluation

Modify `src/evaluate.py` to:
- Add additional metrics
- Change visualization style
- Add per-class metrics

## Paper

This repository supports the paper submission:

**Interpretable Word-level Context-based Sentiment Analysis**

Chenyu Yang<sup>1</sup>, Eric Larson<sup>2</sup>, and Jing Cao<sup>3,*</sup>

<sup>1</sup>Department of Statistics and Data Science, Southern Methodist University, U.S.A  
<sup>2</sup>Department of Computer Science, Southern Methodist University, U.S.A  
<sup>3</sup>Department of Statistics and Data Science, Southern Methodist University, U.S.A

### Abstract

We propose a fine-grained attention-based multiple instance classification (FAMIC) model for interpretable word-level sentiment analysis (SA) using only document-level sentiment labels. By operating at the word level, FAMIC enhances interpretability while maintaining competitive performance in document-level classification. The model generates interpretable outputs such as contextual weighting, word neutrality, and negation cues, offering insights into how context shapes sentiment and how the model arrives at its predictions. FAMIC is built on a straightforward yet effective architecture that combines a multiple instance classification framework with self-attention and positionally encoded self-attention blocks. This design enables the model to capture both local and global contextual dependencies, supporting nuanced sentiment interpretation. We evaluate FAMIC on two sentiment classification datasets and provide an extensive analysis of its interpretability and performance.

**Keywords**: interpretable sentiment analysis; multiple instance classification; self-attention; relative positional embedding

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository was created with assistance from [Cursor](https://cursor.sh), an AI-powered code editor.

## Contact

For questions, issues, or collaboration inquiries, please contact:

- **Chenyu Yang**: chenyuy@smu.edu

For general questions about the repository, please open an issue on [GitHub](https://github.com/YCY198888/FAMIC/issues).
