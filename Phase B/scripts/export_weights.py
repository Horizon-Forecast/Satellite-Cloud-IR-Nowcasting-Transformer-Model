"""
Export trained checkpoints as half-precision (FP16) weights, small enough to commit to git
(each ~58 MB, under GitHub's 100 MB/file limit). This is how the two delivered models in
weights/ were produced from their full-precision training checkpoints.

Inference already runs in FP16 (load_model_for_inference(..., fp16=True)), so casting the
stored weights to FP16 is lossless for how the model is used. Only the model state's
floating-point tensors are cast, model_config and other metadata are preserved.

This is a maintenance tool — the delivered weights are already committed, so it is only
needed after retraining. Point it at the checkpoints a training run produced (see the
Training section of the Maintenance Guide):

    python scripts/export_weights.py CASCADE_CKPT ABLATION_CKPT

If a source checkpoint is not found it is skipped, so partial runs are fine.
"""
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def export(src: str, dst_rel: str) -> None:
    src_path, dst = Path(src), ROOT / dst_rel
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    if not src_path.exists():
        print(f"[skip] source checkpoint not found: {src}")
        return
    ckpt = torch.load(src_path, map_location="cpu", weights_only=True)
    ckpt["model_state"] = {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in ckpt["model_state"].items()
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"[ok] {src} -> {dst_rel}  ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export trained checkpoints to committed FP16 weights.")
    ap.add_argument("cascade_ckpt", nargs="?", default="checkpoints/h5_phase2/gpu0_best.pt",
                    help="Full-precision cascade checkpoint (from the Phase-2 training run).")
    ap.add_argument("ablation_ckpt", nargs="?", default="checkpoints/ablation/gpu0_best.pt",
                    help="Full-precision ablation checkpoint (Phase-2 run with --no-cascade).")
    args = ap.parse_args()
    export(args.cascade_ckpt, "weights/driver_first.pt")
    export(args.ablation_ckpt, "weights/end_to_end.pt")


if __name__ == "__main__":
    main()
