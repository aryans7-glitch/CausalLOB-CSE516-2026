# Original frozen CausalLOB checkpoints

These 11 parts are byte-for-byte slices of `CausalLOB_FROZEN_REPRO_EXPORT.zip`, copied from the original user-provided Google Drive export. The original archive is 201,392,753 bytes and has SHA-256 `f8ca4d6e25ef48fd1265af349cb354e0291f063b541b4c58b9009b5563889f22`.

From `M4_G9/`, execute:

```bash
python Code/src/restore_original_checkpoints.py
python Code/src/verify_frozen_checkpoints.py
```

To put the original files into their expected Colab/Drive directory, append `--extract-root /content/drive/MyDrive/CausalLOB` to the first command. The restorer refuses to replace preexisting different files. File names, byte lengths and hashes of every part are in `PARTS_MANIFEST.json`; hashes for each of the 244 original internal files are contained in the reconstructed ZIP as `FROZEN_CHECKPOINT_MANIFEST.json`.

Splitting into parts smaller than GitHub's browser upload limit does **not** modify the contents of the source archive.
