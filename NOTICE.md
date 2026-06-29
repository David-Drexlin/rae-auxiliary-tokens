# Notice

This code supplement is based on the MIT-licensed RAE implementation. The original MIT license is included as `LICENSE` and must be retained.

Some source files also contain their own copyright or license headers inherited from upstream components. Those headers have been preserved where present.

The following materials are not included and must be obtained separately under their own licenses or terms:

- ImageNet, ImageNet-R, NINCO, Places365, and any other datasets.
- DINOv2, DINOv3, MAE, SigLIP2, LPIPS/VGG, discriminator, and other pretrained weights.
- guided-diffusion evaluation scripts.
- FD-DINOv2 evaluation scripts.
- Slurm, CUDA, PyTorch, torchvision, timm, transformers, and other runtime dependencies.

For a double-blind NeurIPS submission, local machine paths in copied configs/scripts were sanitized to placeholders such as `RAE_ROOT_PLACEHOLDER`, `DATASETS_ROOT_PLACEHOLDER`, and `HOME_PLACEHOLDER`. Replace these placeholders locally before execution.

