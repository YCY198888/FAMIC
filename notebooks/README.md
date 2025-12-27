# FAMIC Notebooks

This folder contains Jupyter notebooks for working with the FAMIC model.

## Notebooks

### `famic_tutorial.ipynb`

A comprehensive tutorial notebook that demonstrates:

1. **Loading Datasets**
   - Downloading datasets from HuggingFace
   - Viewing dataset statistics
   - Switching between Twitter and Wine datasets

2. **Loading Tokenizers**
   - Loading dataset-specific tokenizers
   - Tokenizing sample text
   - Understanding tokenizer vocabulary

3. **Initializing Models**
   - Creating embedding matrices
   - Initializing individual model blocks
   - Initializing the complete FAMIC model
   - Testing model forward pass

## Setup

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install Jupyter:
   ```bash
   pip install jupyter ipykernel
   ```

### Running the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/famic_tutorial.ipynb`

3. Run cells sequentially or use "Run All" from the Cell menu

## Notes

- The notebook automatically handles dataset and tokenizer downloads
- Files are cached in `../data/` and `../data/tokenizers/` directories
- Make sure you have internet access for the first run (to download datasets)
- The notebook is designed to work with both Twitter and Wine datasets

## Troubleshooting

- **Import errors**: Make sure you're running the notebook from the project root or that the `src/` directory is in your Python path
- **Download errors**: Check your internet connection and HuggingFace access
- **CUDA errors**: The notebook will automatically fall back to CPU if CUDA is not available

