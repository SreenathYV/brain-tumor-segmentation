from pathlib import Path
import io

import pyarrow.parquet as pq
from PIL import Image
import matplotlib.pyplot as plt 

data_dir = Path(r"C:\Users\yvsre\Desktop\CV_Project\Dataset")
files = sorted(data_dir.glob("train-*.parquet"))

samples = []
for file in files:
    table = pq.read_table(file, columns=["image", "mask", "label_name", "pid"])
    for row in table.to_pylist()[:2]:
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
        mask = Image.open(io.BytesIO(row["mask"]["bytes"])).convert("L")
        samples.append((image, mask, row["label_name"], row["pid"]))
        if len(samples) == 4:
            break
    if len(samples) == 4:
        break

fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
for i, (image, mask, tumor_type, pid) in enumerate(samples):
    axes[i, 0].imshow(image, cmap="gray")
    axes[i, 0].set_title(f"MRI — {tumor_type}, patient {pid}")
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Tumor mask")
    axes[i, 2].imshow(image, cmap="gray")
    axes[i, 2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[i, 2].set_title("Overlay")
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("sample_preview.png", dpi=150)
plt.show()