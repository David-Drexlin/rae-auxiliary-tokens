# Legacy experiment configurations

These configurations belong to the earlier auxiliary-token research snapshot.
Do not treat their defaults as the current arXiv paper's exact reproduction
recipe. Some launchers override YAML values, and the recovery configs include
older CLS-plus-register targets rather than only four registers.

The supplied reconstruction supplement is in [`paper/`](../paper/), where the
trainer uses an external JSON configuration with schema
`register-decoder-training-v1`. That experiment-specific JSON and its assets are
not included in this repository. See the [release status](../paper/RELEASE_STATUS.md).
