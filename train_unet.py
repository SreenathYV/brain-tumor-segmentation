from pathlib import Path
import io
import random
import shutil

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Paths and settings
DATA_DIR = Path(r"C:\Users\yvsre\Desktop\CV_Project\Dataset")
OUTPUT_DIR = Path(r"C:\Users\yvsre\Desktop\CV_Project\work")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42


def read_training_rows():
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train-*.parquet files found in {DATA_DIR}")

    rows = []
    for file in files:
        table = pq.read_table(file, columns=["image", "mask", "pid"])
        rows.extend(table.to_pylist())

    if not rows:
        raise RuntimeError("No training rows were read from the Parquet files.")

    return rows


def split_by_patient(rows):
    patients = sorted({row["pid"] for row in rows})
    random.Random(SEED).shuffle(patients)

    n_val = max(1, round(len(patients) * 0.2))
    val_patients = set(patients[:n_val])

    train_rows = [row for row in rows if row["pid"] not in val_patients]
    val_rows = [row for row in rows if row["pid"] in val_patients]

    overlap = {row["pid"] for row in train_rows} & {
        row["pid"] for row in val_rows
    }
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {len(overlap)} IDs")

    return train_rows, val_rows


class TumorDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        # Decode MRI while preserving its original bit depth.
        with Image.open(io.BytesIO(row["image"]["bytes"])) as im:
            image = np.asarray(im).astype(np.float32)

        # Remove any invalid pixel values before computing percentiles.
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = image[image > 0]

        if nonzero.size:
            low, high = np.percentile(nonzero, [1, 99])
        else:
            low, high = 0.0, 1.0

        scale = max(float(high - low), 1e-6)
        image = np.clip((image - float(low)) / scale, 0.0, 1.0)

        # Resize MRI smoothly.
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(image_pil, dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Decode and binarize tumor mask; nearest-neighbor keeps labels intact.
        with Image.open(io.BytesIO(row["mask"]["bytes"])) as im:
            mask = np.asarray(im)

        mask = (mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
        mask = (np.asarray(mask_pil) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)

        if not torch.isfinite(image_tensor).all():
            raise ValueError(f"Non-finite MRI values at dataset row {index}")
        if not torch.isfinite(mask_tensor).all():
            raise ValueError(f"Non-finite mask values at dataset row {index}")

        return image_tensor, mask_tensor


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = double_conv(1, 16)
        self.enc2 = double_conv(16, 32)
        self.enc3 = double_conv(32, 64)
        self.enc4 = double_conv(64, 128)
        self.bottleneck = double_conv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = double_conv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = double_conv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Return logits; sigmoid is used for Dice calculation.
        return self.output(d1)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits.float())
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def batch_dice(logits, targets):
    probs = torch.sigmoid(logits.float())
    preds = (probs >= 0.5).float()
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)

    scores = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        (2.0 * intersection + 1e-6) / (denominator + 1e-6),
    )
    return scores.mean().item()


def main():
    # Preserve the original 10-epoch checkpoint once before starting this run.
    baseline_checkpoint = OUTPUT_DIR / "baseline_10_epochs.pt"
    original_checkpoint = OUTPUT_DIR / "best_unet.pt"
    if not baseline_checkpoint.exists() and original_checkpoint.exists():
        shutil.copy2(original_checkpoint, baseline_checkpoint)
        print(f"Preserved baseline checkpoint: {baseline_checkpoint}")
    elif baseline_checkpoint.exists():
        print(f"Keeping existing baseline checkpoint: {baseline_checkpoint}")
    else:
        print("Warning: work\\best_unet.pt was not found; no baseline copy was made.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print("Device: CUDA")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    rows = read_training_rows()
    train_rows, val_rows = split_by_patient(rows)

    train_patients = {row["pid"] for row in train_rows}
    val_patients = {row["pid"] for row in val_rows}

    print(f"Training slices: {len(train_rows)}")
    print(f"Validation slices: {len(val_rows)}")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Patient overlap: {bool(train_patients & val_patients)}")

    train_loader = DataLoader(
        TumorDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TumorDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    checkpoint_path = OUTPUT_DIR / "unet_20_epochs.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0

        for batch_num, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if not torch.isfinite(images).all():
                raise RuntimeError(f"Non-finite image input in batch {batch_num}")
            if not torch.isfinite(masks).all():
                raise RuntimeError(f"Non-finite mask input in batch {batch_num}")

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite model output in epoch {epoch}, batch {batch_num}"
                )

            bce_value = bce(logits.float(), masks.float())
            dice_value = dice_loss(logits, masks)
            loss = bce_value + dice_value

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in epoch {epoch}, batch {batch_num}: "
                    f"BCE={bce_value.item()}, Dice={dice_value.item()}"
                )

            loss.backward()

            # Limit unusually large gradients to improve training stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()

            if epoch == 1 and batch_num == 1:
                print(
                    "First batch check: "
                    f"image range [{images.min().item():.3f}, "
                    f"{images.max().item():.3f}], "
                    f"mask range [{masks.min().item():.1f}, "
                    f"{masks.max().item():.1f}], "
                    f"BCE={bce_value.item():.4f}, "
                    f"Dice loss={dice_value.item():.4f}"
                )

        model.eval()
        val_dice_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)

                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite model output during validation.")

                val_dice_total += batch_dice(logits, masks)

        avg_train_loss = train_loss_total / max(len(train_loader), 1)
        avg_val_dice = val_dice_total / max(len(val_loader), 1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {avg_train_loss:.4f} | "
            f"validation Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "best_val_dice": best_val_dice,
                    "epoch": epoch,
                    "epochs_configured": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            print(f"Saved best model to: {checkpoint_path}")

    print(f"Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()

