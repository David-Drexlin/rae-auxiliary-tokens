# RAE_ROOT_PLACEHOLDER/src/sample_ddp.py
from __future__ import annotations

import os
import sys
import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from omegaconf import OmegaConf

# Make repo imports work when launched from anywhere
REPO_SRC = Path(__file__).resolve().parent
sys.path.append(str(REPO_SRC))

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from utils.condition_sampler import (
    build_condition_spec_from_tcga_csv,
    build_cartesian_condition_spec,
    build_condition_pool,
    shard_pool_for_rank,
    compute_total_samples_rounded,
    class_frequency_from_condition_spec,
    build_class_only_pool,
    ConditionPool,
)

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.state_utils import (
    decode_stage2_state,
    duplicate_state_for_guidance,
    final_state_from_trajectory,
    infer_aux_state_spec,
    make_initial_sample_state,
    split_guided_state,
    state_float,
)
from stage2.transport import create_transport, Sampler
import json
import csv

class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # append mode so resume appends new records
        self.f = open(self.path, "a", buffering=1)

    def write(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()


def write_manifest_info(out_folder: Path, info: Dict[str, Any]) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    p = out_folder / "manifest_info.json"
    with open(p, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def merge_rank_manifests(
    out_folder: Path,
    world_size: int,
    meta_fields: List[str],
) -> None:
    """
    Merge manifest_rankXXX.jsonl -> manifest.jsonl + manifest.csv (streaming, no big RAM usage).
    """
    out_folder = Path(out_folder)
    merged_jsonl = out_folder / "manifest.jsonl"
    merged_csv = out_folder / "manifest.csv"

    # CSV header
    meta_cols = [f"meta_{f}" for f in meta_fields]
    header = ["index", "filename", "y"] + meta_cols + ["cond_mode", "rank", "seed"]

    # Merge JSONL by concatenation + simultaneously write CSV.
    with open(merged_jsonl, "w") as f_jsonl, open(merged_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=header)
        writer.writeheader()

        total_rows = 0
        for r in range(world_size):
            rp = out_folder / f"manifest_rank{r:03d}.jsonl"
            if not rp.exists():
                continue
            with open(rp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    f_jsonl.write(line + "\n")

                    rec = json.loads(line)

                    row = {
                        "index": int(rec["index"]),
                        "filename": str(rec["filename"]),
                        "y": int(rec["y"]),
                        "cond_mode": str(rec.get("cond_mode", "")),
                        "rank": int(rec.get("rank", -1)),
                        "seed": int(rec.get("seed", -1)),
                    }
                    # meta columns (if present)
                    meta = rec.get("meta", None)
                    if meta is None:
                        for c in meta_cols:
                            row[c] = ""
                    else:
                        # meta is list[int] length F
                        for f, v in zip(meta_fields, meta):
                            row[f"meta_{f}"] = int(v)

                    writer.writerow(row)
                    total_rows += 1

        print(f"[manifest] merged {total_rows} rows -> {merged_csv} and {merged_jsonl}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def create_npz_from_sample_folder(sample_dir: str, num: int, out_name: str = "samples.npz") -> str:
    """
    Packs {sample_dir}/*.png into ADM-style NPZ: key arr_0, dtype uint8, shape (N,H,W,3).
    WARNING: loads everything into RAM.
    """
    folder = Path(sample_dir)
    pngs = sorted(folder.glob("*.png"))
    if len(pngs) == 0:
        raise RuntimeError(f"No PNGs found in {sample_dir}")

    if num is not None and len(pngs) < num:
        print(f"[warn] requested num={num} but found only {len(pngs)} pngs. Packing available files.")

    pngs = pngs[:num] if (num is not None) else pngs

    first = np.array(Image.open(pngs[0]).convert("RGB"), dtype=np.uint8)
    H, W, _ = first.shape
    arr = np.empty((len(pngs), H, W, 3), dtype=np.uint8)
    arr[0] = first

    for i, p in enumerate(tqdm(pngs[1:], desc="Packing NPZ", unit="img"), start=1):
        arr[i] = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

    out_path = folder / out_name
    np.savez_compressed(out_path, arr_0=arr)
    print(f"[npz] wrote {out_path}  shape={arr.shape} dtype={arr.dtype}")
    return str(out_path)


def save_latent_shard(
    out_folder: Path,
    *,
    latent_subdir: str,
    rank: int,
    step_idx: int,
    indices,
    y,
    meta,
    state,
) -> None:
    latent_dir = out_folder / latent_subdir
    ensure_dir(latent_dir)

    payload: Dict[str, Any] = {
        "indices": torch.as_tensor(indices, dtype=torch.int64),
        "y": torch.as_tensor(y, dtype=torch.int64),
    }
    if meta is not None:
        payload["meta"] = torch.as_tensor(meta, dtype=torch.int64)

    if torch.is_tensor(state):
        payload["z"] = state.detach().to("cpu", dtype=torch.float16)
    else:
        z, aux = state
        payload["z"] = z.detach().to("cpu", dtype=torch.float16)
        payload["aux"] = aux.detach().to("cpu", dtype=torch.float16)

    shard_path = latent_dir / f"rank{rank:03d}_step{step_idx:06d}.pt"
    torch.save(payload, shard_path)


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    import json
    with open(p, "r") as f:
        return json.load(f)


def load_stage2_weights(model: torch.nn.Module, ckpt_path: str, use_ema: bool, rank: int) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ema"] if use_ema else ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[ckpt] loaded {ckpt_path} (ema={use_ema}) missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"[ckpt] missing keys (first 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"[ckpt] unexpected keys (first 10): {unexpected[:10]}")
    model.eval()
    return ckpt


def get_guidance_params(guidance_cfg: Dict[str, Any]):
    def guidance_value(key: str, default: float):
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    scale = float(guidance_cfg.get("scale", 1.0))
    aux_scale = float(guidance_cfg.get("aux_scale", guidance_cfg.get("aux-scale", 0.0)))
    method = str(guidance_cfg.get("method", "cfg"))
    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    return scale, aux_scale, method, t_min, t_max


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict update: values from src override dst.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_sample_config(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    d = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(d, dict):
        raise ValueError("sample-config must resolve to a mapping.")
    return d


def apply_cli_overrides(base: Dict[str, Any], cli: argparse.Namespace) -> Dict[str, Any]:
    """
    Only overrides keys for CLI args that were explicitly provided (i.e., not None).
    """
    overrides: Dict[str, Any] = {}

    def set_if_not_none(key_path: str, value):
        if value is None:
            return
        # write nested mapping
        cur = overrides
        parts = key_path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    # stage2 pointers
    set_if_not_none("stage2.config", getattr(cli, "config", None))
    set_if_not_none("stage2.ckpt", getattr(cli, "ckpt", None))
    set_if_not_none("stage2.use_ema", getattr(cli, "use_ema", None))

    # output/run
    set_if_not_none("output.sample_dir", getattr(cli, "sample_dir", None))
    set_if_not_none("output.save_folder", getattr(cli, "save_folder", None))
    set_if_not_none("output.pack_npz", getattr(cli, "pack_npz", None))

    set_if_not_none("run.num_samples", getattr(cli, "num_samples", None))
    set_if_not_none("run.per_proc_batch_size", getattr(cli, "per_proc_batch_size", None))
    set_if_not_none("run.precision", getattr(cli, "precision", None))
    set_if_not_none("run.tf32", getattr(cli, "tf32", None))
    set_if_not_none("run.global_seed", getattr(cli, "global_seed", None))

    # conditioning
    label_sampling = getattr(cli, "label_sampling", None)
    if label_sampling is not None:
        cond_mode = "equal_class" if label_sampling == "equal" else "random_class"
        set_if_not_none("conditioning.mode", cond_mode)
    set_if_not_none("conditioning.mode", getattr(cli, "cond_mode", None))
    set_if_not_none("conditioning.csv_path", getattr(cli, "csv_path", None))
    set_if_not_none("conditioning.image_root", getattr(cli, "image_root", None))
    set_if_not_none("conditioning.count_level", getattr(cli, "count_level", None))
    set_if_not_none("conditioning.id_field", getattr(cli, "id_field", None))
    set_if_not_none("conditioning.label_field", getattr(cli, "label_field", None))
    set_if_not_none("conditioning.slide_sep", getattr(cli, "slide_sep", None))

    set_if_not_none("conditioning.y_vocab_json", getattr(cli, "y_vocab_json", None))
    set_if_not_none("conditioning.meta_vocabs_json", getattr(cli, "meta_vocabs_json", None))

    set_if_not_none("conditioning.skip_unk_y", getattr(cli, "skip_unk_y", None))
    set_if_not_none("conditioning.drop_meta_unk", getattr(cli, "drop_meta_unk", None))
    set_if_not_none("conditioning.skip_unk_meta", getattr(cli, "skip_unk_meta", None))

    set_if_not_none("conditioning.cartesian_max_tuples", getattr(cli, "cartesian_max_tuples", None))

    return deep_update(base, overrides)


def require(cfg: Dict[str, Any], key_path: str):
    cur: Any = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required sample-config key: '{key_path}'")
        cur = cur[k]
    return cur


def infer_skip_ids_for_class_only(
    *,
    skip_unk_y: bool,
    num_classes: int,
    null_label: int,
    y_vocab: Optional[Dict[str, Any]],
) -> list[int]:
    """
    Infer ids to exclude for class-only sampling modes without assuming that
    class 0 is automatically an unknown label.
    """
    if not skip_unk_y:
        return []

    skip_ids: set[int] = set()

    if 0 <= int(null_label) < int(num_classes):
        skip_ids.add(int(null_label))

    if isinstance(y_vocab, dict):
        unk_keys = {"unk", "unknown", "__unk__", "[unk]", "<unk>"}
        for raw_key, raw_value in y_vocab.items():
            key = str(raw_key).strip().lower()
            if key not in unk_keys:
                continue
            try:
                idx = int(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < int(num_classes):
                skip_ids.add(idx)

    return sorted(skip_ids)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(cfg: Dict[str, Any]):
    # -------------------------
    # DDP + device
    # -------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    # -------------------------
    # Read sample-config
    # -------------------------
    stage2_yaml = str(require(cfg, "stage2.config"))
    ckpt_path = str(require(cfg, "stage2.ckpt"))
    use_ema = bool(require(cfg, "stage2.use_ema"))

    sample_dir = Path(str(require(cfg, "output.sample_dir")))
    output_cfg = cfg.get("output", {})
    save_folder = output_cfg.get("save_folder", None)
    pack_npz = bool(output_cfg.get("pack_npz", True))
    save_latent_shards = bool(output_cfg.get("save_latent_shards", False))
    latent_shard_dir = str(output_cfg.get("latent_shard_dir", "latent_shards"))

    num_samples = int(require(cfg, "run.num_samples"))
    per_proc_batch_size = int(require(cfg, "run.per_proc_batch_size"))
    precision = str(require(cfg, "run.precision")).lower()
    tf32 = bool(require(cfg, "run.tf32"))
    global_seed = int(require(cfg, "run.global_seed"))

    cond = cfg.get("conditioning", {})
    cond_mode = str(cond.get("mode", "random_class"))
    csv_path = cond.get("csv_path", None)
    image_root = cond.get("image_root", None)
    count_level = str(cond.get("count_level", "image"))
    id_field = str(cond.get("id_field", "slide_submitter_id"))
    label_field = str(cond.get("label_field", "cancer_type"))
    slide_sep = str(cond.get("slide_sep", "__"))

    y_vocab_json = cond.get("y_vocab_json", None)
    meta_vocabs_json = cond.get("meta_vocabs_json", None)
    skip_unk_y = bool(cond.get("skip_unk_y", True))
    drop_meta_unk = bool(cond.get("drop_meta_unk", False))
    skip_unk_meta = bool(cond.get("skip_unk_meta", True))
    cartesian_max_tuples = cond.get("cartesian_max_tuples", None)
    cartesian_max_tuples = int(cartesian_max_tuples) if cartesian_max_tuples is not None else None

    if precision not in {"fp32", "bf16"}:
        raise ValueError("run.precision must be fp32 or bf16 for this sampler.")

    # -------------------------
    # Seed / TF32 / autocast
    # -------------------------
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_grad_enabled(False)

    use_bf16 = (precision == "bf16")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 but device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
        print(f"stage2_yaml={stage2_yaml}")
        print(f"ckpt={ckpt_path} use_ema={use_ema}")
        print(f"cond_mode={cond_mode}")

    # -------------------------
    # Load stage-2 YAML + parse sections (order: data first)
    # -------------------------
    stage2_cfg = OmegaConf.load(stage2_yaml)
    (
        data_config,
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        eval_config,
    ) = parse_configs(stage2_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Stage-2 YAML must provide stage_1 and stage_2 sections.")

    data_cfg = {} if data_config is None else dict(OmegaConf.to_container(data_config, resolve=True))
    misc = {} if misc_config is None else dict(OmegaConf.to_container(misc_config, resolve=True))
    transport_cfg = {} if transport_config is None else dict(OmegaConf.to_container(transport_config, resolve=True))
    sampler_cfg = {} if sampler_config is None else dict(OmegaConf.to_container(sampler_config, resolve=True))
    guidance_cfg = {} if guidance_config is None else dict(OmegaConf.to_container(guidance_config, resolve=True))

    sampler_override = cfg.get("sampler_override", None)
    if sampler_override is not None:
        sampler_override = OmegaConf.to_container(OmegaConf.create(sampler_override), resolve=True)
        if not isinstance(sampler_override, dict):
            raise ValueError("sampler_override must resolve to a mapping.")
        sampler_cfg = deep_update(sampler_cfg, sampler_override)

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    # -------------------------
    # Instantiate models
    # -------------------------
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    aux_state_spec = infer_aux_state_spec(rae, latent_size=latent_size)

    target_stage1_cfg = cfg.get("target_stage1", {}) or {}
    target_stage1_config_path = str(target_stage1_cfg.get("config_path", "")).strip()
    target_stage1_decoder_ckpt = target_stage1_cfg.get("decoder_ckpt", None)
    decode_cfg = cfg.get("decode", {}) or {}
    use_aux_for_decode = bool(decode_cfg.get("use_aux_for_decode", True))

    decode_rae = rae
    if target_stage1_config_path:
        from sample_recovered_aux_ddp import (
            build_stage1_from_source_config,  # local import to avoid circular import at module import time
        )

        decode_rae = build_stage1_from_source_config(
            target_stage1_config_path,
            device,
            decoder_ckpt=target_stage1_decoder_ckpt,
        )

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()

    ckpt = load_stage2_weights(model, ckpt_path, use_ema=use_ema, rank=rank)

    # -------------------------
    # Transport / sampler
    # -------------------------
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)

    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode == "ODE":
        sample_fn = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        sample_fn = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampler.mode={sampler_mode}")

    # -------------------------
    # Guidance
    # -------------------------
    guidance_scale, guidance_scale_aux, guidance_method, t_min, t_max = get_guidance_params(guidance_cfg)
    using_cfg = guidance_scale > 1.0

    guid_model_forward = None
    if using_cfg and guidance_method == "autoguidance":
        guid_model_config = guidance_cfg.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("guidance_model must be provided for autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    # -------------------------
    # Conditioning availability
    # -------------------------
    meta_fields = list(data_cfg.get("meta_fields", [])) if isinstance(data_cfg.get("meta_fields", []), list) else []
    model_has_meta = hasattr(model, "meta_embedders") and getattr(model, "meta_embedders") is not None
    meta_enabled = (len(meta_fields) > 0) and model_has_meta

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))

    if rank == 0:
        print(f"meta_fields(from stage2 yaml)={meta_fields}")
        print(f"model_has_meta={model_has_meta} -> meta_enabled={meta_enabled}")
        print(f"num_classes={num_classes} null_label={null_label}")
        print(f"latent_size={latent_size} time_dist_shift={time_dist_shift:.4f}")
        print(f"aux_state_mode={aux_state_spec.mode} aux_state_shape={aux_state_spec.shape}")
        print(
            f"guidance: patch_scale={guidance_scale} aux_scale={guidance_scale_aux} "
            f"method={guidance_method} interval=({t_min},{t_max})"
        )
        print(f"sampler_mode={sampler_mode} sampler_params={sampler_params}")

    # -------------------------
    # Output folder
    # -------------------------
    ensure_dir(sample_dir)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    sampling_method = sampler_params.get("sampling_method", "na")
    num_steps = sampler_params.get("num_steps", sampler_params.get("steps", "na"))
    base_folder = f"{ckpt_name}-cfg{guidance_scale:.2f}-bs{per_proc_batch_size}-{sampler_mode}-{num_steps}-{sampling_method}-{precision}"
    folder_name = str(save_folder) if save_folder else base_folder

    out_folder = sample_dir / folder_name
    if rank == 0:
        ensure_dir(out_folder)
        print(f"Saving PNG samples at {out_folder}")
    dist.barrier()

    # -------------------------
    # DDP layout
    # -------------------------
    n = per_proc_batch_size
    global_bs = n * world_size
    total_samples = compute_total_samples_rounded(num_samples, world_size, n)
    per_rank_samples = total_samples // world_size
    iterations = per_rank_samples // n

    if rank == 0:
        print(f"num_samples(requested)={num_samples} -> total_samples(rounded)={total_samples}")
        print(f"per_rank_samples={per_rank_samples} iterations={iterations} global_bs={global_bs}")

    # -------------------------
    # Load vocabs if needed
    # -------------------------
    # Try: ckpt -> explicit path -> auto from ckpt experiment dir
    y_vocab = ckpt.get("y_vocab", None) or load_json(y_vocab_json)

    if y_vocab is None:
        ckpt_p = Path(ckpt_path).resolve()
        # common layouts:
        #   <exp>/checkpoints/ep-xxxx.pt  -> want <exp>/y_vocab.json
        candidates = [
            ckpt_p.parent / "y_vocab.json",         # <...>/checkpoints/y_vocab.json (if you ever put it there)
            ckpt_p.parent.parent / "y_vocab.json",  # <exp>/y_vocab.json  (this is what you want)
        ]
        for c in candidates:
            if c.exists():
                if rank == 0:
                    print(f"[vocab] auto-loading y_vocab from {c}")
                y_vocab = load_json(str(c))
                break
    meta_vocabs = ckpt.get("meta_vocabs", None) or load_json(meta_vocabs_json)

    joint_requested = cond_mode in {"uniform_observed", "actual_joint", "cartesian"}
    if joint_requested and not meta_enabled:
        raise ValueError(
            f"cond_mode='{cond_mode}' requires meta_fields non-empty AND model trained with meta_num_classes. "
            f"meta_fields={meta_fields}, model_has_meta={model_has_meta}."
        )

    # -------------------------
    # Build rank-local condition tensors (y, meta)
    # -------------------------
    rank_view_y: torch.Tensor
    rank_view_meta: Optional[torch.Tensor]

    if cond_mode == "uncond":
        y_pool = np.full((total_samples,), int(null_label), dtype=np.int64)
        meta_pool = np.zeros((total_samples, len(meta_fields)), dtype=np.int64) if meta_enabled else None
        pool = ConditionPool(y=y_pool, meta=meta_pool)
        rv = shard_pool_for_rank(pool, rank=rank, world_size=world_size, per_rank_samples=per_rank_samples, batch_size=n)
        rank_view_y, rank_view_meta = rv.y, rv.meta

    elif cond_mode in {"random_class", "equal_class", "actual_class"}:
        if cond_mode == "actual_class":
            if y_vocab is None:
                raise ValueError("actual_class requires y_vocab (checkpoint or conditioning.y_vocab_json).")
            if csv_path is None:
                raise ValueError("actual_class requires conditioning.csv_path.")
            if count_level == "image" and image_root is None:
                raise ValueError("count_level=image requires conditioning.image_root.")

            spec = build_condition_spec_from_tcga_csv(
                csv_path=csv_path,
                image_root=image_root if count_level == "image" else None,
                slide_sep=slide_sep,
                id_field=id_field,
                label_field=label_field,
                meta_fields=[],
                y_vocab=y_vocab,
                meta_vocabs={},
                count_level=count_level,
                skip_unk_y=skip_unk_y,
            )
            class_ids, class_probs = class_frequency_from_condition_spec(spec)
            pool = build_class_only_pool(
                num_classes=num_classes,
                total_samples=total_samples,
                seed=global_seed,
                mode="actual",
                class_ids=class_ids,
                class_probs=class_probs,
                skip_ids=[],
            )

        elif cond_mode == "random_class":
            skip = infer_skip_ids_for_class_only(
                skip_unk_y=skip_unk_y,
                num_classes=num_classes,
                null_label=null_label,
                y_vocab=y_vocab,
            )
            pool = build_class_only_pool(
                num_classes=num_classes,
                total_samples=total_samples,
                seed=global_seed,
                mode="random",
                skip_ids=skip,
            )

        else:
            valid = list(range(num_classes))
            for skip_id in infer_skip_ids_for_class_only(
                skip_unk_y=skip_unk_y,
                num_classes=num_classes,
                null_label=null_label,
                y_vocab=y_vocab,
            ):
                if skip_id in valid:
                    valid.remove(skip_id)
            if len(valid) == 0:
                raise RuntimeError("No valid classes remain for equal_class.")
            if num_samples % len(valid) != 0:
                raise ValueError(f"equal_class requires num_samples divisible by #valid_classes ({len(valid)}).")

            reps = num_samples // len(valid)
            base = np.repeat(np.array(valid, dtype=np.int64), reps)
            rng = np.random.default_rng(global_seed)
            rng.shuffle(base)

            if total_samples > len(base):
                tail = rng.choice(np.array(valid, dtype=np.int64), size=(total_samples - len(base)), replace=True)
                y_pool = np.concatenate([base, tail], axis=0)
            else:
                y_pool = base[:total_samples]

            pool = ConditionPool(y=y_pool.astype(np.int64), meta=None)

        rv = shard_pool_for_rank(pool, rank=rank, world_size=world_size, per_rank_samples=per_rank_samples, batch_size=n)
        rank_view_y, rank_view_meta = rv.y, None

    else:
        # joint modes
        if y_vocab is None or meta_vocabs is None:
            raise ValueError(
                f"cond_mode='{cond_mode}' requires y_vocab and meta_vocabs. "
                f"Provide checkpoint fields or conditioning.y_vocab_json/meta_vocabs_json."
            )

        if cond_mode in {"uniform_observed", "actual_joint"}:
            if csv_path is None:
                raise ValueError(f"{cond_mode} requires conditioning.csv_path.")
            if count_level == "image" and image_root is None:
                raise ValueError("count_level=image requires conditioning.image_root.")

            spec = build_condition_spec_from_tcga_csv(
                csv_path=csv_path,
                image_root=image_root if count_level == "image" else None,
                slide_sep=slide_sep,
                id_field=id_field,
                label_field=label_field,
                meta_fields=meta_fields,
                y_vocab=y_vocab,
                meta_vocabs=meta_vocabs,
                count_level=count_level,
                skip_unk_y=skip_unk_y,
                drop_rows_with_any_meta_unk=drop_meta_unk,
            )
            pool_mode = "uniform_observed" if cond_mode == "uniform_observed" else "actual"
            pool = build_condition_pool(spec, mode=pool_mode, total_samples=total_samples, seed=global_seed)

        else:
            spec = build_cartesian_condition_spec(
                y_vocab=y_vocab,
                meta_vocabs=meta_vocabs,
                meta_fields=meta_fields,
                num_classes=num_classes,
                null_label=null_label,
                skip_unk_y=skip_unk_y,
                skip_unk_meta=skip_unk_meta,
                skip_cfg_null_meta=True,
                max_tuples=cartesian_max_tuples,
                seed=global_seed,
            )
            pool = build_condition_pool(spec, mode="cartesian_uniform", total_samples=total_samples, seed=global_seed)

        rv = shard_pool_for_rank(pool, rank=rank, world_size=world_size, per_rank_samples=per_rank_samples, batch_size=n)
        rank_view_y, rank_view_meta = rv.y, rv.meta

    if rank == 0:
        print(f"[conds] y shape={tuple(rank_view_y.shape)} meta={'None' if rank_view_meta is None else tuple(rank_view_meta.shape)}")

    # -------------------------
    # Resume-safe skipping (synchronize already across ranks)
    # -------------------------
    already_tensor = torch.zeros(1, device=device, dtype=torch.long)
    if rank == 0:
        existing_pngs = list(out_folder.glob("*.png"))
        already = (len(existing_pngs) // global_bs) * global_bs
        already_tensor[0] = already
        if already > 0:
            print(f"[resume] found {len(existing_pngs)} pngs, using already={already} (multiple of global_bs={global_bs})")
    dist.broadcast(already_tensor, src=0)
    already = int(already_tensor.item())

    skip_iters = already // global_bs
    if skip_iters > iterations:
        raise RuntimeError(f"[resume] skip_iters={skip_iters} > iterations={iterations}. Folder/config mismatch?")

    if rank == 0:
        print(f"[resume] skip_iters={skip_iters} of iterations={iterations}")
    dist.barrier()

    # -------------------------
    # Manifest writers
    # -------------------------
    manifest_rank_path = out_folder / f"manifest_rank{rank:03d}.jsonl"
    mw = JsonlWriter(manifest_rank_path)

    if rank == 0:
        info = {
            "stage2_yaml": stage2_yaml,
            "ckpt_path": ckpt_path,
            "use_ema": use_ema,
            "cond_mode": cond_mode,
            "meta_fields": meta_fields,
            "conditioning": {
                "csv_path": csv_path,
                "image_root": image_root,
                "count_level": count_level,
                "id_field": id_field,
                "label_field": label_field,
                "slide_sep": slide_sep,
                "y_vocab_json": y_vocab_json,
                "meta_vocabs_json": meta_vocabs_json,
                "skip_unk_y": skip_unk_y,
                "drop_meta_unk": drop_meta_unk,
                "skip_unk_meta": skip_unk_meta,
                "cartesian_max_tuples": cartesian_max_tuples,
            },
            "num_samples_requested": num_samples,
            "total_samples_rounded": total_samples,
            "per_proc_batch_size": per_proc_batch_size,
            "world_size": world_size,
            "precision": precision,
            "tf32": tf32,
            "global_seed": global_seed,
            "guidance_scale": guidance_scale,
            "guidance_aux_scale": guidance_scale_aux,
            "guidance_method": guidance_method,
            "cfg_interval": [t_min, t_max],
            "sampler_mode": sampler_mode,
            "sampler_params": sampler_params,
            "latent_size": list(latent_size),
            "time_dist_shift": float(time_dist_shift),
            "target_stage1_config": target_stage1_config_path or None,
            "target_stage1_decoder_ckpt": target_stage1_decoder_ckpt,
            "use_aux_for_decode": use_aux_for_decode,
        }
        write_manifest_info(out_folder, info)

    # -------------------------
    # Sampling loop (skip already-done iterations)
    # -------------------------
    start_index_base = already
    total_written = start_index_base
    pbar_range = range(skip_iters, iterations)
    pbar = tqdm(pbar_range, desc="Sampling", unit="iter") if rank == 0 else pbar_range

    try:
        for step_idx in pbar:
            with autocast(**autocast_kwargs):
                state = make_initial_sample_state(
                    n=n,
                    latent_size=latent_size,
                    aux_state_spec=aux_state_spec,
                    device=device,
                )

                # record conditional y/meta (before CFG concat)
                y_cond = rank_view_y[step_idx].to(device, non_blocking=True)
                meta_cond = rank_view_meta[step_idx].to(device, non_blocking=True) if rank_view_meta is not None else None

                model_fn = model.forward
                model_kwargs: Dict[str, Any] = {"y": y_cond}
                if meta_cond is not None:
                    model_kwargs["meta"] = meta_cond

                if using_cfg:
                    state = duplicate_state_for_guidance(state)

                    y_null = torch.full((n,), int(null_label), device=device, dtype=y_cond.dtype)
                    y_in = torch.cat([y_cond, y_null], dim=0)

                    model_kwargs = {
                        "y": y_in,
                        "cfg_scale": float(guidance_scale),
                        "cfg_scale_aux": float(guidance_scale_aux),
                        "cfg_interval": (float(t_min), float(t_max)),
                    }

                    if meta_cond is not None:
                        meta_null = torch.zeros_like(meta_cond)
                        meta_in = torch.cat([meta_cond, meta_null], dim=0)
                        model_kwargs["meta"] = meta_in

                    if guidance_method == "autoguidance":
                        if guid_model_forward is None:
                            raise RuntimeError("Guidance model forward is not initialized.")
                        model_kwargs["additional_model_forward"] = guid_model_forward
                        model_fn = model.forward_with_autoguidance
                    else:
                        model_fn = model.forward_with_cfg

                sampled_state = final_state_from_trajectory(
                    sample_fn(state, model_fn, **model_kwargs)
                )
                if using_cfg:
                    sampled_state = split_guided_state(sampled_state)

                sampled_state = state_float(sampled_state)
                imgs = decode_stage2_state(
                    decode_rae,
                    sampled_state,
                    use_aux_for_decode=use_aux_for_decode,
                ).clamp(0, 1)
                imgs = imgs.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # save + manifest rows
            y_cpu = y_cond.detach().cpu().long().numpy()
            meta_cpu = meta_cond.detach().cpu().long().numpy() if meta_cond is not None else None

            indices = [total_written + local_idx * world_size + rank for local_idx in range(len(imgs))]
            if save_latent_shards:
                save_latent_shard(
                    out_folder,
                    latent_subdir=latent_shard_dir,
                    rank=rank,
                    step_idx=step_idx,
                    indices=indices,
                    y=y_cpu,
                    meta=meta_cpu,
                    state=sampled_state,
                )

            for local_idx, sample in enumerate(imgs):
                index = indices[local_idx]
                filename = f"{index:06d}.png"
                Image.fromarray(sample).save(out_folder / filename)

                mw.write({
                    "index": int(index),
                    "filename": filename,
                    "y": int(y_cpu[local_idx]),
                    "meta": (meta_cpu[local_idx].tolist() if meta_cpu is not None else None),
                    "cond_mode": cond_mode,
                    "rank": int(rank),
                    "seed": int(seed),
                })

            total_written += global_bs
            dist.barrier()

    finally:
        mw.close()

    dist.barrier()

    # -------------------------
    # Merge manifests + optional NPZ pack (rank0)
    # -------------------------
    if rank == 0:
        merge_rank_manifests(out_folder, world_size=world_size, meta_fields=meta_fields)
        if pack_npz:
            create_npz_from_sample_folder(str(out_folder), num=num_samples, out_name="samples.npz")
        print("Done.")

    dist.barrier()
    dist.destroy_process_group()
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Stage-2 DDP sampler with sample-config YAML support")

    # New preferred entry
    p.add_argument("--sample-config", type=str, default=None, help="Sampling YAML (preferred).")

    # Backward-compatible direct args (also work as overrides when sample-config is used)
    p.add_argument("--config", type=str, default=None, help="Stage-2 training YAML (override stage2.config).")
    p.add_argument("--ckpt", type=str, default=None, help="Stage-2 checkpoint (.pt) (override stage2.ckpt).")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--sample-dir", type=str, default=None)
    p.add_argument("--save-folder", type=str, default=None)
    p.add_argument("--pack-npz", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--per-proc-batch-size", type=int, default=None)
    p.add_argument("--global-seed", type=int, default=None)
    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default=None)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--label-sampling", type=str, choices=["equal", "random"], default=None)

    p.add_argument(
        "--cond-mode",
        type=str,
        choices=["uncond", "random_class", "equal_class", "actual_class", "uniform_observed", "actual_joint", "cartesian"],
        default=None,
    )
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--image-root", type=str, default=None)
    p.add_argument("--count-level", type=str, choices=["image", "slide"], default=None)

    p.add_argument("--id-field", type=str, default=None)
    p.add_argument("--label-field", type=str, default=None)
    p.add_argument("--slide-sep", type=str, default=None)

    p.add_argument("--y-vocab-json", type=str, default=None)
    p.add_argument("--meta-vocabs-json", type=str, default=None)

    p.add_argument("--skip-unk-y", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--drop-meta-unk", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--skip-unk-meta", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cartesian-max-tuples", type=int, default=None)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.sample_config is None:
        # Backward compatibility: require config+ckpt
        if args.config is None or args.ckpt is None:
            raise SystemExit("Either pass --sample-config, or pass both --config and --ckpt.")
        # construct cfg dict from CLI directly
        cfg = {
            "stage2": {"config": args.config, "ckpt": args.ckpt, "use_ema": True if args.use_ema is None else bool(args.use_ema)},
            "output": {
                "sample_dir": args.sample_dir or "samples",
                "save_folder": args.save_folder,
                "pack_npz": True if args.pack_npz is None else bool(args.pack_npz),
                "save_latent_shards": False,
                "latent_shard_dir": "latent_shards",
            },
            "run": {
                "num_samples": int(args.num_samples or 50_000),
                "per_proc_batch_size": int(args.per_proc_batch_size or 64),
                "precision": str(args.precision or "bf16"),
                "tf32": True if args.tf32 is None else bool(args.tf32),
                "global_seed": int(args.global_seed or 0),
            },
            "conditioning": {
                "mode": str(
                    args.cond_mode
                    or ("equal_class" if args.label_sampling == "equal" else None)
                    or ("random_class" if args.label_sampling == "random" else None)
                    or "random_class"
                ),
                "csv_path": args.csv_path,
                "image_root": args.image_root,
                "count_level": args.count_level or "image",
                "id_field": args.id_field or "slide_submitter_id",
                "label_field": args.label_field or "cancer_type",
                "slide_sep": args.slide_sep or "__",
                "y_vocab_json": args.y_vocab_json,
                "meta_vocabs_json": args.meta_vocabs_json,
                "skip_unk_y": True if args.skip_unk_y is None else bool(args.skip_unk_y),
                "drop_meta_unk": False if args.drop_meta_unk is None else bool(args.drop_meta_unk),
                "skip_unk_meta": True if args.skip_unk_meta is None else bool(args.skip_unk_meta),
                "cartesian_max_tuples": args.cartesian_max_tuples,
            },
        }
    else:
        # Preferred: load sample yaml then apply CLI overrides
        cfg = load_sample_config(args.sample_config)
        cfg = apply_cli_overrides(cfg, args)

    main(cfg)
