# Source release status

Updated: 14 October 2026.

This directory contains the supplied reconstruction-training source supplement for
**Decoding Register Tokens Preserves Global Scene Details** (arXiv:2610.13299).
It is a source release, not a claim that every experiment in the paper is
reproducible from the repository alone.

## Import provenance

The 33 original files (29 Python files, three shell scripts, and one README) were
imported without changing their contents. The shell launchers are executable.
This status document was added separately; it is not part of the original archive.

- Source archive SHA-256: `3a68ec5f23076d7026ae2a44807fa029f44c564d3472aa426e32a7136d36fd44`.
- Git tree of the 33 imported files, before this status document was added:
  `247c97e860dfa8db4700ef024bc379eb06d5be99`.
- Previous public repository commit:
  `4998a6e5109f76ca0e59c4199f1db30c2debe0a7`.

The imported Git tree was compared with a tree calculated from the extracted
archive, verifying the file contents and paths, with executable modes for the
three shell launchers. The existing root-level experimental source, configs,
and launchers were not numerically modified by this import.

## Included

The package contains the frozen encoder interface, patch-only and four-register
joint-self-attention decoders, image reconstruction losses, distributed training,
EMA/checkpoint handling, input validation, and reconstruction diagnostics.
The package restricts native auxiliary inputs to four register outputs, excluding
the encoder CLS output. A separate learned decoder class token is retained.
The existing double addition of learned register-slot positions is preserved.

The main entry point is `code/train.py`. It accepts the external configuration
schema `register-decoder-training-v1`. The two scripts select either no native
register inputs or four native register inputs, with a 16-epoch endpoint.

## External inputs and scope limits

This package does not include experiment-specific training JSONs, encoder/decoder
configuration assets, pretrained encoder/discriminator/perceptual weights,
normalization statistics, image manifests, trained decoder checkpoints, or a
current four-register MHAP/generation reproduction package. The older recovery
and generation experiments in the root-level source are legacy code, not a
replacement for the missing current configurations.

A usable training JSON must bind the appropriate local source directory, model
parameters, training settings, train/monitor/final manifests, asset paths, and
expected checksums. The source root for this package is
`${PACKAGE_ROOT}/model_src`, where `${PACKAGE_ROOT}` resolves to `paper/code`.
Do not substitute a legacy YAML or guess scientific asset checksums.

The packaged validation requires disjoint train/monitor/final image membership
and exactly 50,000 final-evaluation images. Adapting the package to another
dataset requires an explicit protocol change. The generic encoder wrapper does
not itself pin a remote model revision; use the intended local snapshot and
record its assets/checksums in the supplied configuration.

## Validation performed for this import

- All 29 Python files pass Python syntax parsing (`ast.parse`).
- All three shell launchers pass `bash -n`.
- The training entry point and all three shell launchers return successfully
  for `--help`.
- The 33-file Git tree matches the extracted source archive as described above.

These checks do not execute a model forward pass. No asset-backed input check,
GPU training, distributed resume test, checkpoint compatibility test, or paper
metric reproduction was performed as part of this repository update. Required
model/data assets were not supplied. A complete reproduction release needs the
corresponding experiment inputs and an end-to-end validation record.

## Licensing

Preserve the repository `LICENSE` and `NOTICE.md`, and the third-party licensing
and source attribution retained in `code/model_src/__init__.py`. Dataset and
pretrained-weight terms are separate from source-code terms. Those assets are
not redistributed in this directory.
