"""Extract Step-3.7-Flash vision weights from upstream bf16 shards into an
MLX-compatible safetensors file with the NHWC Conv2d transpose applied.

Why this script: MLX `Conv2d` expects NHWC layout — kernel shape
(out_channels, kernel_h, kernel_w, in_channels). PyTorch's NCHW convention
gives (out, in, kH, kW). The conv1 patch-embed + 2 vit_downsampler kernels
must be pre-transposed at extract time (or at load time via sanitize); we
do it at extract so the resulting shard loads cleanly through stock
mlx_vlm load_weights with no special-case logic.

Also CRITICAL: `mx.load + mx.save_safetensors` round-trips lazy values as
all-zero data. Always `mx.eval(v)` each tensor before storing it in the
output dict (see https://github.com/ml-explore/mlx/issues — exact bug
caught in this project's commit history).

Usage:
    python extract_vision_weights.py \\
        --src /path/to/Step-3.7-Flash \\
        --dst /path/to/your-quantized-model-dir/model-00023-of-00023.safetensors

Pulls 666 `vision_model.*` keys + 1 `vit_large_projector.weight` key.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


CONV2D_KEY_HINTS = ("conv1.", "vit_downsampler1.", "vit_downsampler2.")


def needs_nchw_to_nhwc(key: str, ndim: int) -> bool:
    """True if this weight tensor is a Conv2d kernel that PyTorch stored
    NCHW (out_C, in_C, kH, kW) and MLX wants NHWC (out, kH, kW, in)."""
    return ndim == 4 and key.endswith(".weight") and any(h in key for h in CONV2D_KEY_HINTS)


def extract(src_dir: Path, dst_path: Path) -> None:
    # Find all upstream vision shards (model-vit-*.safetensors files)
    vit_shards = sorted(src_dir.glob("model-vit-*.safetensors"))
    if not vit_shards:
        raise FileNotFoundError(
            f"No model-vit-*.safetensors found in {src_dir}. "
            "Download stepfun-ai/Step-3.7-Flash first (hf download stepfun-ai/Step-3.7-Flash --local-dir <dst>)."
        )
    print(f"Found {len(vit_shards)} upstream vision shards:")
    for s in vit_shards:
        print(f"  {s.name} ({s.stat().st_size / 1e9:.2f} GB)")

    out: dict[str, mx.array] = {}
    n_transposed = 0
    n_kept = 0
    for shard in vit_shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if not (key.startswith("vision_model.") or key.startswith("vit_large_projector")):
                    continue
                # Pull as torch tensor → numpy float32 (numpy doesn't have bf16)
                t = f.get_tensor(key)
                arr = t.to(__import__("torch").float32).numpy()
                if needs_nchw_to_nhwc(key, arr.ndim):
                    arr = arr.transpose(0, 2, 3, 1)
                    n_transposed += 1
                mx_t = mx.array(arr).astype(mx.bfloat16)
                # *** CRITICAL: must eval before save or the file is all zeros ***
                mx.eval(mx_t)
                out[key] = mx_t
                n_kept += 1
    print(f"\nKept {n_kept} keys; transposed {n_transposed} Conv2d kernels NCHW→NHWC.")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # metadata={'format': 'mlx'} is what mlx-vlm load_weights checks
    mx.save_safetensors(str(dst_path), out, metadata={"format": "mlx"})
    print(f"Wrote {dst_path} ({dst_path.stat().st_size / 1e9:.2f} GB)")
    print("Verify with:")
    print(f"  python -c \"from safetensors import safe_open; "
          f"f = safe_open('{dst_path}', framework='pt'); "
          f"print(len(list(f.keys())), 'keys')\"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Path to extracted upstream stepfun-ai/Step-3.7-Flash dir (containing model-vit-*.safetensors)")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output safetensors path inside your quantized model dir")
    args = ap.parse_args()
    extract(args.src.expanduser().resolve(), args.dst.expanduser().resolve())


if __name__ == "__main__":
    main()
