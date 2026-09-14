# Code

## Main Notebook
`CausalLOB_M1_FROM_SCRATCH_COLAB.ipynb`

## How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Run the notebook from top to bottom.
4. On the first run, FI-2010 is downloaded and preprocessing checkpoints are created.
5. On later runs, keep:

```python
FORCE_REBUILD = False


Checkpoint Location

/content/drive/MyDrive/CausalLOB/M1_G9/checkpoints/
