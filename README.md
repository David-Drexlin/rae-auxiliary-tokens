# Code Supplement: When do Representation Autoencoders Need Auxiliary Tokens?

This folder is a minimal code supplement for the paper experiments. It is derived from the MIT-licensed RAE codebase and contains the paper-specific code needed for:

- Stage-1 RAE decoder training and reconstruction evaluation.
- Auxiliary-token decoder interfaces: prepending, cross-attention, AdaLN-style modulation, global-only controls, learned-token controls, PatchMAP controls, and recovered DINO auxiliary tokens.
- DINOv2, DINOv2 without registers, DINOv3, MAE, and SigLIP2 ImageNet decoder configs.
- Post-hoc DINOv2 CLS-plus-register prediction from patch tokens.
- Patch-only Stage-2 DiT/DDT training, sampling, AutoGuidance evaluation, and recovered-auxiliary decoding.
- Probe and diagnostic scripts used for complementarity, semantic retention, and summary-token controls.

## What is intentionally not included

The supplement excludes large or non-redistributable artifacts:

- ImageNet / ImageNet-R / NINCO / Places365 data.
- Model checkpoints, generated samples, logs, caches, NPZ/NPY feature stores, and paper figures.
- The Apptainer/Singularity container image.
- Third-party evaluation repos such as guided-diffusion and FD-DINOv2.
- Downloaded pretrained weights for DINOv2, DINOv3, MAE, SigLIP2, VGG, LPIPS, or discriminator backbones.

Configs and Slurm scripts may contain placeholder paths such as `RAE_ROOT_PLACEHOLDER`, `DATASETS_ROOT_PLACEHOLDER`, and `HOME_PLACEHOLDER`. Replace these with local paths before running.

## Layout

- `src/`: Python source for Stage-1, Stage-2, evaluation, auxiliary prediction, and diagnostics.
- `configs/decoder/`: ViT decoder architecture configs.
- `configs/register_prediction/`: MHAP patch-to-auxiliary predictor configs.
- `configs/stage1/training/ImageNet/`: ImageNet100 Stage-1 configs and controls.
- `configs/stage1/training/ImageNet1k/`: ImageNet1K Stage-1 scale-up configs.
- `configs/stage2/training/ImageNet256/`: selected ImageNet Stage-2 training configs.
- `configs/stage2/sampling/ImageNet256/`: selected ImageNet Stage-2 sampling/evaluation configs.
- `*.sh`: selected launch/evaluation wrappers for the reported experiment families.
- `requirements.txt`, `environment.yml`: dependency snapshots.
- `LICENSE`, `NOTICE.md`: licensing and redistribution notes.

## Paper-to-Code Map

- Main Stage-1 decoder training uses `src/train_stage1.py`, `src/stage1/`, the `run_stage1_*_imagenet100_array.sh` launchers, and configs under `configs/stage1/training/ImageNet/`.
- DINOv2 auxiliary interfaces are implemented in `src/stage1/decoders/decoder.py` and `src/stage1/rae.py`; the corresponding configs include `DINO_decB_Patch+CLS_prepend.yaml`, `DINO_decB_Patch+Register_prepend.yaml`, `DINO_decB_Patch+Register+CLS_prepend.yaml`, `DINO_decB_Patch+Register+CLS_CA.yaml`, and `DINO_decB_Patch+Register+CLS_AdaLN.yaml`.
- MAE and SigLIP2 contrast rows use the matching `MAE_decB_*` and `SigLIP2_decB_*` configs under `configs/stage1/training/ImageNet/`.
- Global-only controls use `run_stage1_dino_global_imagenet100_array.sh`, `run_stage1_mae_imagenet100_array.sh`, `run_stage1_siglip2_imagenet100_array.sh`, and the `Global*` / `GlobalOnly*` Stage-1 configs.
- Probe diagnostics use `run_aux_complementarity_probe.sh`, `run_eval_stage1_semantic_retention.sh`, `run_summarize_stage1_semantic_retention_dino.sh`, `run_global_color_hist_probe.sh`, `run_cls_register_fusion_probe.sh`, `run_patchmap_summary_probe.sh`, and the corresponding `src/analyze_*` scripts.
- ImageNet1K Stage-1 scale-up uses `run_dino_imagenet1k_stats_array.sh`, `run_stage1_dino_imagenet1k_8gpu_single.sh`, and configs under `configs/stage1/training/ImageNet1k/`.
- OOD reconstruction evaluation uses `stage1_eval_OOD_DINO_single.sh`, `src/stage1_sample_ddp.py`, and `src/eval_stage1_recon.py`.
- Post-hoc auxiliary prediction uses `cache_dino_dataset_features.sh`, `run_register_prediction_in100.sh`, `src/register_prediction/`, and configs under `configs/register_prediction/`.
- Stage-2 patch-only DiT/DDT training uses `train_ImageNet_DiT.sh`, `src/train.py`, `src/stage2/`, and configs under `configs/stage2/training/ImageNet256/`.
- Stage-2 sampling/evaluation uses `src/sample_ddp.py`, `src/sample_recovered_aux_ddp.py`, `run_stage2_in1k_class_sample.sh`, `stage2_eval_ImageNet1k*.sh`, `submit_stage2_dino_imagenet1k_patch80_recovery_chain_fixv3_40gb.sh`, and `submit_stage2_in1k_ag150_eval_chains.sh`.

## License Notes

The included code is distributed under the MIT license from the upstream RAE repository. Keep `LICENSE` with any copy of this supplement. Paper-specific modifications in this bundle should be treated as MIT-compatible unless the authors choose a different license before public release.

Dataset, pretrained-weight, and external-evaluator licenses are separate. Do not redistribute ImageNet or downloaded model weights inside the NeurIPS supplement.
