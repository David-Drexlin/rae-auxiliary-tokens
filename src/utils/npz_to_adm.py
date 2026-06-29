#!/usr/bin/env python3
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--key", default=None, help="Key to read from input NPZ. If unset, auto-pick.")
    args = ap.parse_args()

    z = np.load(args.in_npz)
    keys = list(z.keys())

    key = args.key
    if key is None:
        # common candidates
        for cand in ["arr_0", "images", "imgs", "x"]:
            if cand in keys:
                key = cand
                break
        if key is None:
            raise SystemExit(f"Could not auto-pick key from {keys}. Pass --key explicitly.")

    arr = z[key]

    # ADM evaluator expects uint8 NHWC in [0,255]
    if arr.dtype != np.uint8:
        # try common float formats
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0,255).astype(np.uint8)
        else:
            arr = arr.clip(0,255).astype(np.uint8)

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise SystemExit(f"Expected NHWC with 3 channels, got shape {arr.shape}")

    np.savez(args.out_npz, arr_0=arr)
    print(f"Wrote {args.out_npz} with key arr_0, shape={arr.shape}, dtype={arr.dtype}")

if __name__ == "__main__":
    main()
