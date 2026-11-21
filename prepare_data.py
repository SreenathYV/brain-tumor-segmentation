from pathlib import Path
import io
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; saves the image without opening a plot window
import matplotlib.pyplot as plt

DATA_DIR = Path(r"C:\Users\yvsre\Desktop\CV_Project\Dataset")
OUTPUT_DIR = Path(r"C:\Users\yvsre\Desktop\CV_Project\work")
OUTPUT_DIR.mkdir(exist_ok=True)

# Read only the training shards. Keep the separate test shard untouched.
rows = []
for file in sorted(DATA_DIR.glob("train-*.parquet")):
    table = pq.read_table(file, columns=["image", "mask", "pid", "label_name"])
    rows.extend(table.to_pylist())

# Split by patient, so slices from one patient cannot cross the split.
patient_ids = sorted({row["pid"] for row in rows})
random.Random(42).shuffle(patient_ids)
n_val = max(1, round(len(patient_ids) * 0.2))
val_patients = set(patient_ids[:n_val])

train_rows = [r for r in rows if r["pid"] not in val_patients]
val_rows = [r for r in rows if r["pid"] in val_patients]

print(f"Patients: {len(patient_ids)} total")
print(f"Train: {len(train_rows)} slices from {len(patient_ids) - n_val} patients")
print(f"Validation: {len(val_rows)} slices from {n_val} patients")
print("Patient overlap:", bool(
    {r["pid"] for r in train_rows} & {r["pid"] for r in val_rows}
))

def decode_and_resize(row, size=(256, 256)):
    # Preserve the MRI's original bit depth.
    image = np.array(Image.open(io.BytesIO(row["image"]["bytes"]))).astype(np.float32)

    # Normalize each slice using its nonzero 1st and 99th intensity percentiles.
    nonzero = image[image > 0]
    low, high = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)

    # Resize the image smoothly.
    image_pil = Image.fromarray(image, mode="F").resize(
        size, Image.Resampling.BILINEAR
    )
    image = np.array(image_pil, dtype=np.float32)

    # Make a binary mask and resize with nearest-neighbor interpolation.
    mask = Image.open(io.BytesIO(row["mask"]["bytes"]))
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)
    mask_pil = Image.fromarray(mask * 255).resize(
        size, Image.Resampling.NEAREST
    )
    mask = (np.array(mask_pil) > 0).astype(np.uint8)

    return image, mask

# Save a normalized preview from validation patients.
samples = val_rows[:4]
fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
if len(samples) == 1:
    axes = np.array([axes])

for i, row in enumerate(samples):
    image, mask = decode_and_resize(row)
    axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_title(f"MRI — {row['label_name']}, patient {row['pid']}")
    axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title("Binary mask")
    axes[i, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[i, 2].imshow(mask, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
    axes[i, 2].set_title("Overlay")
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
preview_path = OUTPUT_DIR / "normalized_preview.png"
plt.savefig(preview_path, dpi=150)
print(f"Saved preview: {preview_path}")
# plt.show()