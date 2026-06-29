#!/usr/bin/env python3
"""Create matched MHAP attention-map comparisons for DINO, MAE, and SigLIP2."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SOURCES = {
    "DINO": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/dino_patch_to_dino_regcls_in100/runs/mhap/attention_maps/val"),
    "MAE": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/mae_to_dino_regcls_in100/runs/mhap/attention_maps/val"),
    "SigLIP2": Path("RAE_ROOT_PLACEHOLDER/assets/analysis/siglip2_to_dino_regcls_in100/runs/mhap/attention_maps/val"),
}

OUT_DIR = Path("RAE_ROOT_PLACEHOLDER/assets/analysis/mhap_attention_comparison")
NUM_IMAGES = 5
PADDING = 24
HEADER_H = 72
BG = (255, 255, 255)
FG = (20, 20, 20)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = load_font(42)
    label_font = load_font(28)

    sample_names = [f"attn_{idx:03d}.png" for idx in range(NUM_IMAGES)]
    per_encoder = {name: [root / sample for sample in sample_names] for name, root in SOURCES.items()}

    # Copy the individual originals into a dedicated folder for convenience.
    for encoder, paths in per_encoder.items():
        enc_dir = OUT_DIR / encoder.lower()
        enc_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dst = enc_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    # Build one comparison strip per sample and one stacked contact sheet.
    first = Image.open(per_encoder["DINO"][0]).convert("RGB")
    w, h = first.size
    strip_w = len(SOURCES) * w + (len(SOURCES) + 1) * PADDING
    strip_h = h + HEADER_H + 2 * PADDING

    strips = []
    for sample_name in sample_names:
        canvas = Image.new("RGB", (strip_w, strip_h), BG)
        draw = ImageDraw.Draw(canvas)
        x = PADDING
        for encoder in SOURCES:
            draw.text((x, PADDING), encoder, fill=FG, font=title_font)
            img = Image.open(SOURCES[encoder] / sample_name).convert("RGB")
            canvas.paste(img, (x, HEADER_H))
            draw.text((x, HEADER_H + h + 8), sample_name.replace(".png", ""), fill=FG, font=label_font)
            x += w + PADDING
        out_path = OUT_DIR / f"compare_{sample_name}"
        canvas.save(out_path)
        strips.append(canvas)

    contact_h = len(strips) * strip_h
    contact = Image.new("RGB", (strip_w, contact_h), BG)
    y = 0
    for strip in strips:
        contact.paste(strip, (0, y))
        y += strip_h
    contact.save(OUT_DIR / "comparison_contact_sheet.png")

    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
