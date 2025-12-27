# HuggingFace Authentication Guide

If you encounter a **401 Unauthorized** error when downloading datasets or tokenizers, you need to authenticate with HuggingFace.

## Quick Fix

### Option 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:HF_TOKEN='your_huggingface_token_here'
```

**Windows Command Prompt:**
```cmd
set HF_TOKEN=your_huggingface_token_here
```

**Linux/Mac:**
```bash
export HF_TOKEN=your_huggingface_token_here
```

### Option 2: Python Code

In your Python script or notebook:
```python
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
```

Or use the helper function:
```python
from src.datasets import authenticate_huggingface
authenticate_huggingface(token='your_huggingface_token_here')
```

### Option 3: HuggingFace CLI

```bash
huggingface-cli login
```

This will prompt you to enter your token interactively.

## Getting Your Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or use an existing one)
3. Copy the token (starts with `hf_...`)
4. Use it in one of the methods above

## Important Notes

- **Restart your kernel/script** after setting the environment variable if you're using a Jupyter notebook
- The token is automatically used by the download functions
- Never commit your token to git (it's already in `.gitignore`)

## Troubleshooting

### Still getting 401 errors?

1. **Verify the token is set:**
   ```python
   import os
   print(os.getenv('HF_TOKEN'))  # Should show your token
   ```

2. **Check token permissions:**
   - Make sure your token has "read" access
   - If the repository is private, ensure you have access to it

3. **Try logging in via CLI:**
   ```bash
   huggingface-cli login
   ```

4. **Check repository access:**
   - Verify you have access to `ycy198888/jds_support_files`
   - The repository might be private or gated

### Path Issues

If you see errors about file paths, make sure you've:
1. **Restarted your kernel** (in Jupyter) to pick up code changes
2. The code now uses `datasets/twitter_cleaned2024.csv` instead of just `twitter_cleaned2024.csv`

To restart Jupyter kernel: `Kernel → Restart Kernel` or press `0` twice in command mode.

