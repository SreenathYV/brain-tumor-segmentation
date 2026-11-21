# Brain Tumor Segmentation

A coursework project for **Deep Learning with Computer Vision**. It trains a 2D U-Net to segment tumor regions in brain MRI slices and includes a Streamlit interface for interactive inference.

## Features

- Patient-disjoint training and validation split
- Binary tumor-mask segmentation with a 2D U-Net
- 10-epoch baseline and 20-epoch extended experiment
- Streamlit app to upload a 2D MRI image and view/download the predicted mask
- Overall and per-tumor Dice and IoU evaluation

## Dataset

This project uses the Figshare Brain Tumor dataset from Cheng et al. (3,064 contrast-enhanced T1 MRI slices from 233 patients, with binary tumor masks and labels for glioma, meningioma, and pituitary tumors).

The dataset files are not included in this repository. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.1512427) or the source used for your Parquet copy, then place the three `train-*.parquet` shards and `test-00000-of-00001.parquet` in a `Dataset/` folder at the project root.

Citation: Cheng, J. et al. (2015). “Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition.” *PLOS ONE*, 10(10), e0140381. DOI: 10.1371/journal.pone.0140381.

## Model and results

Training uses 256 x 256 images, batch size 4, Adam with learning rate 0.0001, and binary cross-entropy plus Dice loss. The 20-epoch run achieved a best validation Dice of 0.7640 and mean held-out test Dice of 0.7241 (mean IoU 0.6280).

| Tumor type | 10-epoch Dice | 20-epoch Dice | Test slices |
|---|---:|---:|---:|
| Glioma | 0.5576 | 0.6150 | 245 |
| Meningioma | 0.7722 | 0.8408 | 112 |
| Pituitary | 0.7880 | 0.7980 | 185 |

The 20-epoch checkpoint is included at `work/unet_20_epochs.pt` so the demo can run without training first. The original 10-epoch baseline and other generated outputs are not included.

## Setup

Use Python 3.10 or later. Create and activate a virtual environment if desired, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build appropriate for your NVIDIA driver from the [official PyTorch selector](https://pytorch.org/get-started/locally/). The app also runs on CPU.

## Run the web app

From the project root:

```powershell
python -m streamlit run app.py
```

The app accepts a single 2D PNG, JPG, or TIFF slice. It does not accept full 3D NIfTI volumes or DICOM studies. This is an academic demonstration, not a medical diagnostic tool.

## Reproduce training and evaluation

The Parquet data must be in `Dataset/` as described above.

```powershell
python train_unet.py
python evaluate.py --checkpoint unet_20_epochs.pt
```

The training script writes checkpoints and previews to `work/`. The evaluator uses the held-out test shard and reports results overall and by tumor type.