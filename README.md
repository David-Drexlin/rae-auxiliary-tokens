# Decoding Register Tokens Preserves Global Scene Details

David Jacob Drexlin, Julius Hense, and Jonas Dippel  
[Paper: arXiv:2610.13299](https://arxiv.org/abs/2610.13299)

> **Release status (14 October 2026): source-only, not a complete reproduction release.**
> The supplied reconstruction-training supplement is in [`paper/`](paper/).
> The root-level `src/`, `configs/`, and shell launchers are an **older research snapshot**; their defaults must not be treated as the current paper's reproduction recipe.
> Trained checkpoints, experiment-specific training JSONs, manifests, normalization statistics, and a current four-register MHAP/generation package are **not included**. Training and paper metrics have not been reproduced as part of this repository update.

## Start here

- **Current supplied reconstruction supplement:** [`paper/README.md`](paper/README.md).
- **Register-aware decoder:** [`paper/code/model_src/stage1/decoders/decoder.py`](paper/code/model_src/stage1/decoders/decoder.py).
- **Encoder features and token selection:** [`paper/code/model_src/stage1/encoders/dinov2.py`](paper/code/model_src/stage1/encoders/dinov2.py) and [`paper/code/model_src/stage1/rae.py`](paper/code/model_src/stage1/rae.py).
- **Training entry point and input schema:** [`paper/code/train.py`](paper/code/train.py) and [`paper/code/trainer/protocol.py`](paper/code/trainer/protocol.py).
- **Import provenance and validation scope:** [`paper/RELEASE_STATUS.md`](paper/RELEASE_STATUS.md).
- **Historical code documentation:** [`docs/LEGACY_README.md`](docs/LEGACY_README.md).

The `paper/` package keeps all spatial patches and optionally prepends four native register outputs for joint self-attention. The encoder CLS output is not a decoder input in this configuration; the decoder's own learned class token is a separate parameter. See the paper and package source for the feature-coordinate and positional-embedding conventions.

## Running the supplied package

Start in `paper/`, not the repository root:

```bash
cd paper
bash scripts/check_inputs.sh /path/to/training.json
bash scripts/train_decoder.sh /path/to/training.json /path/to/patch_output
bash scripts/train_decoder_with_registers.sh /path/to/training.json /path/to/register_output
```

These commands require external inputs. There is no bundled training JSON: the package validates schema `register-decoder-training-v1`, local asset checksums, and image manifests. Read the package README before launching. A successful input check does not establish forward-pass correctness or reproduce a paper result.

The two training launchers select patch-only or patch-plus-four-register inputs. The supplied trainer fixes the endpoint at 16 epochs. The packaged split validation also expects 50,000 final-evaluation images, so adapting it to another dataset requires an explicitly documented protocol change rather than simply replacing the image folder.

## Legacy snapshot

The existing root-level code is retained at its original paths to avoid silently breaking earlier experiments. It includes broader auxiliary-token ablations and an older five-output CLS-plus-register recovery configuration. Some launchers override YAML settings, and some defaults differ from the principal experiments in the current manuscript. Do not mix these configurations or checkpoints with the `paper/` package without checking compatibility.

The pre-update snapshot is commit `4998a6e5109f76ca0e59c4199f1db30c2debe0a7`. Its original README is preserved in `docs/LEGACY_README.md`.

## Citation

```bibtex
@misc{drexlin2026decoding,
  title={Decoding Register Tokens Preserves Global Scene Details},
  author={David Jacob Drexlin and Julius Hense and Jonas Dippel},
  year={2026},
  eprint={2610.13299},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.13299}
}
```

## Licenses and external assets

Retain `LICENSE`, `NOTICE.md`, and the third-party notices in the supplied source. In particular, the supplement retains third-party licensing text in `paper/code/model_src/__init__.py`. Dataset, pretrained-weight, and external-evaluator terms are separate; those assets are not redistributed here.
