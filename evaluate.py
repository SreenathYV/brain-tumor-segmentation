import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from train_unet import DATA_DIR, OUTPUT_DIR, BATCH_SIZE, TumorDataset, UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved brain tumor segmentation checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="unet_20_epochs.pt",
        help=(
            "Checkpoint filename in the work folder (or a full path). "
            "Defaults to unet_20_epochs.pt."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file = DATA_DIR / "test-00000-of-00001.parquet"

    requested_checkpoint = Path(args.checkpoint)
    checkpoint_file = (
        requested_checkpoint
        if requested_checkpoint.is_absolute()
        else OUTPUT_DIR / requested_checkpoint
    )

    if not test_file.exists():
        raise FileNotFoundError(f"Test shard not found: {test_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")

    table = pq.read_table(
        test_file,
        columns=["image", "mask", "pid", "label_name"],
    )
    test_rows = table.to_pylist()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {checkpoint_file.name}")
    print(f"Test slices: {len(test_rows)}")
    print(f"Test patients: {len({row['pid'] for row in test_rows})}")

    test_loader = DataLoader(
        TumorDataset(test_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet().to(device)
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dice_scores = []
    iou_scores = []
    class_dice = {}
    class_iou = {}
    glioma_previews = []

    with torch.no_grad():
        row_index = 0
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            predictions = (torch.sigmoid(logits) >= 0.5).float()

            for i in range(images.shape[0]):
                row = test_rows[row_index]
                label = row["label_name"]
                pred = predictions[i, 0]
                target = masks[i, 0]
                intersection = (pred * target).sum().item()
                pred_area = pred.sum().item()
                target_area = target.sum().item()
                union = pred_area + target_area - intersection

                dice = (
                    1.0
                    if pred_area + target_area == 0
                    else (2.0 * intersection) / (pred_area + target_area)
                )
                iou = 1.0 if union == 0 else intersection / union

                dice_scores.append(dice)
                iou_scores.append(iou)
                class_dice.setdefault(label, []).append(dice)
                class_iou.setdefault(label, []).append(iou)

                if label == "glioma" and len(glioma_previews) < 8:
                    glioma_previews.append(
                        (
                            images[i, 0].cpu().numpy(),
                            target.cpu().numpy(),
                            pred.cpu().numpy(),
                            row["pid"],
                        )
                    )

                row_index += 1

    print(f"\nOverall test Dice (mean per slice): {np.mean(dice_scores):.4f}")
    print(f"Overall test IoU (mean per slice):  {np.mean(iou_scores):.4f}")
    print("\nTest metrics by tumor type:")
    for label in sorted(class_dice):
        print(
            f"  {label}: Dice={np.mean(class_dice[label]):.4f}, "
            f"IoU={np.mean(class_iou[label]):.4f}, "
            f"slices={len(class_dice[label])}"
        )

    if not glioma_previews:
        print("No glioma slices found; no preview image was created.")
        return

    fig, axes = plt.subplots(
        len(glioma_previews),
        3,
        figsize=(10, 3 * len(glioma_previews)),
        squeeze=False,
    )
    for i, (image, target, pred, pid) in enumerate(glioma_previews):
        axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"MRI - glioma, patient {pid}")
        axes[i, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(pred, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction overlay")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    preview_path = OUTPUT_DIR / f"glioma_predictions_{checkpoint_file.stem}.png"
    plt.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved glioma prediction preview: {preview_path}")


if __name__ == "__main__":
    main()
