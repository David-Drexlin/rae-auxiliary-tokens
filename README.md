# Decoding Register Tokens Preserves Global Scene Details

Decoder training from a frozen vision encoder's spatial patches, with optional
access to four terminal register tokens through joint self-attention.

[Paper](https://arxiv.org/abs/2610.13299)

## Setup

Use a CUDA-enabled PyTorch environment with `torchvision`, `transformers`,
`numpy`, `Pillow`, `requests`, `tqdm`, and `safetensors`.

Supply a training JSON using the `register-decoder-training-v1` schema in
[`code/trainer/protocol.py`](code/trainer/protocol.py). It specifies the model,
training settings, image manifests, local pretrained weights, normalization
statistics, and asset checksums. These inputs and trained checkpoints are
not bundled and must be supplied separately. Set `source_root` to `${PACKAGE_ROOT}/model_src`;
`${PACKAGE_ROOT}` resolves to this repository's `code/` directory.

Image manifests are CSVs with `stable_id`, `path`, `label`, and `sha256` columns;
`parent_id` is optional. The training, monitoring, and final-evaluation image
sets must be disjoint. The final-evaluation manifest contains 50,000 images.

## Train

Run on a node with the GPUs specified by `training.world_size`:

```bash
bash train_decoder.sh /path/to/training.json /path/to/output_patches
bash train_decoder_with_registers.sh /path/to/training.json /path/to/output_registers
```

The scripts select patch-only or patch-plus-four-register input and use the
remaining settings from the same JSON. Both train for 16 epochs. Output
directories must be empty for new runs. To resume, append the corresponding
training checkpoint path as the third argument. Set `PYTHON_BIN` to choose
the Python executable.

To validate input bindings before training:

```bash
python3 code/train.py --config /path/to/training.json \
    --output /path/to/output --register-input native --check-inputs
```

## Acknowledgments and licenses

This implementation builds on [RAE](https://github.com/bytetriper/RAE), with
training ideas also drawn from [RAEv2](https://github.com/nanovisionx/RAEv2).
We thank the upstream authors. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for retained copyright notices, component licenses, and source attribution.
Pretrained weights and datasets have their own terms.
