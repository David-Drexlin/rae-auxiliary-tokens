# Register-aware image reconstruction

Python implementation for training a ViT-B image decoder from frozen spatial
patch features, optionally including four terminal encoder registers through
joint self-attention.
The training endpoint is 16 epochs, with EMA checkpointing and distributed training.

## Contents

- `code/train.py`: training entry point and input validation.
- `code/trainer/`: optimization, checkpointing, and distributed execution.
- `code/model_src/`: encoder interface, decoder, and image-loss components.
- `code/evaluation/`: training diagnostics and reconstruction objective.
- `scripts/check_inputs.sh`: validate the configuration and input bindings.
- `scripts/train_decoder.sh`: train from patches.
- `scripts/train_decoder_with_registers.sh`: train from patches and four registers.

## Environment and inputs

Use a Python environment containing CUDA-enabled PyTorch, torchvision,
transformers, NumPy, Pillow, requests, tqdm, and safetensors. The launch scripts
use `python3`; set `PYTHON_BIN` to select another Python executable.

Supply a training JSON with schema `register-decoder-training-v1`, the model
configuration, image manifests, pretrained weights, and normalization statistics.
These inputs are external to this source archive. Configuration validation is
implemented in `code/trainer/protocol.py`. Paths and expected checksums in the
configuration must identify the supplied local files.

For configurations using environment placeholders, set their corresponding
variables before launching, for example:

```bash
export ASSET_DIR=/path/to/model_assets
export DATA_DIR=/path/to/image_manifests
export TORCH_HOME=/path/to/torch_cache
```

`${PACKAGE_ROOT}` resolves to the extracted `code/` directory. Each image
manifest is a CSV with `stable_id`, `path`, `label`, and `sha256` columns;
`parent_id` is optional. Relative image paths resolve from the manifest directory.

## Validate inputs

```bash
bash scripts/check_inputs.sh /path/to/training.json
```

This checks configuration, declared asset checksums, manifest membership, and
split separation. It does not run a model forward pass or start training.

## Train

Use the same input JSON for both commands and separate output directories:

```bash
bash scripts/train_decoder.sh /path/to/training.json /path/to/output_patches
bash scripts/train_decoder_with_registers.sh /path/to/training.json /path/to/output_registers
```

The launchers differ in one register-input setting. Both use the supplied seed,
image manifests, preprocessing, image losses, optimizers, learning-rate schedule,
EMA settings, batch size, and 16-epoch endpoint. The entry point records the chosen
interface in the run configuration and condition identifier.

Run each command on an allocated node with the GPUs required by
`training.world_size`. One process is started per configured GPU on a single node.
The output directory must be empty for a new run. Intermediate checkpoints do not
stop training. These scripts do not submit scheduler jobs.

To resume, append the saved training checkpoint to the corresponding command:

```bash
bash scripts/train_decoder.sh /path/to/training.json /path/to/output_patches /path/to/checkpoint.pt
bash scripts/train_decoder_with_registers.sh /path/to/training.json /path/to/output_registers /path/to/checkpoint.pt
```

Use a checkpoint from the same run and interface. Resume restores model,
optimizer, scheduler, EMA, RNG, and training-position state. Keep the configuration,
input bindings, source location, and distributed geometry unchanged. Checkpoints
and inference exports are written under the output directory.

Third-party licensing and attribution are retained in the Python source.
