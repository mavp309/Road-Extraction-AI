"""
Inference — Road Segmentation
Loads a trained checkpoint and produces (1, H, W) probability maps.

Usage:
    python infer.py --checkpoint checkpoints/best.pt \
                    --input path/to/image.npz \
                    --output prob_map.npz

    # batch over a metadata CSV
    python infer.py --checkpoint checkpoints/best.pt \
                    --metadata output_dataset/metadata.csv \
                    --output_dir predictions/
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from train import RoadSegFormer   # re-use model definition


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: str, device: torch.device) -> RoadSegFormer:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    # infer backbone from checkpoint if stored, else default to mit-b2
    backbone = ckpt.get("backbone", "nvidia/mit-b2")
    model = RoadSegFormer(backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def preprocess(npz_path: str) -> torch.Tensor:
    """Load a single .npz patch → normalised (1, 3, H, W) tensor."""
    img = np.load(npz_path)["data"]          # (3, H, W) float32 [0,1]
    img = img.transpose(1, 2, 0)             # (H, W, 3)
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)             # (3, H, W)
    return torch.from_numpy(img).unsqueeze(0) # (1, 3, H, W)


@torch.no_grad()
def predict(model: RoadSegFormer, tensor: torch.Tensor,
            device: torch.device) -> np.ndarray:
    """
    Returns (1, H, W) float32 probability map in [0, 1].
    Sigmoid applied here — model outputs raw logits.
    """
    tensor = tensor.to(device)
    logits = model(tensor)                        # (1, 1, H, W)
    prob   = torch.sigmoid(logits)
    return prob.squeeze(0).cpu().numpy()          # (1, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      default=None,
                        help="Single .npz image patch")
    parser.add_argument("--output",     default="prediction.npz",
                        help="Output .npz for single-file mode")
    parser.add_argument("--metadata",   default=None,
                        help="CSV with image_path column for batch mode")
    parser.add_argument("--output_dir", default="predictions/",
                        help="Output dir for batch mode")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Optional: also save binary mask alongside probs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint} on {device}")

    if args.input:
        # ── single file mode ──────────────────
        tensor = preprocess(args.input)
        prob   = predict(model, tensor, device)  # (1, H, W)
        np.savez_compressed(args.output, prob=prob)
        print(f"Saved probability map → {args.output}  shape={prob.shape}")

    elif args.metadata:
        # ── batch mode ────────────────────────
        import pandas as pd
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(args.metadata)
        for i, row in df.iterrows():
            tensor = preprocess(row["image_path"])
            prob   = predict(model, tensor, device)   # (1, H, W)

            stem   = Path(row["image_path"]).stem
            out_path = out_dir / f"{stem}_prob.npz"
            np.savez_compressed(out_path, prob=prob)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(df)} done")

        print(f"Batch inference complete → {out_dir}")
    else:
        print("Provide --input (single) or --metadata (batch).")


if __name__ == "__main__":
    main()
