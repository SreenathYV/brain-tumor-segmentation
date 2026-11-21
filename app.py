import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from train_unet import IMAGE_SIZE, OUTPUT_DIR, UNet


APP_TITLE = "Brain Tumor Segmentation"
CHECKPOINT_PATH = OUTPUT_DIR / "unet_20_epochs.pt"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #14303d;
        --muted: #5b7080;
        --teal: #167d88;
        --teal-dark: #0f5c66;
        --teal-soft: #e6f3f4;
        --line: #dbe5ea;
        --wash: #f2f7f8;
        --good: #1c8a5a;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(180deg, #f4f9fa 0%, #e9f2f4 100%) !important;
        color: var(--ink) !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
    }

    [data-testid="stHeader"] { background: transparent !important; }
    /* Hide Deploy / menu, but keep the button that re-opens the sidebar */
    [data-testid="stToolbarActions"], [data-testid="stDecoration"],
    [data-testid="stMainMenu"], .stAppDeployButton, #MainMenu { display: none !important; }

    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(20, 48, 61, 0.1);
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] * { color: var(--teal-dark) !important; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }

    .brand {
        display: flex; align-items: center; gap: 10px;
        margin: 4px 0 18px;
    }
    .brand-logo {
        width: 40px; height: 40px; border-radius: 12px;
        background: linear-gradient(135deg, #1b97a3, #0f5c66);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 6px 14px rgba(22, 125, 136, 0.3);
    }
    .brand-name { font-weight: 800; font-size: 1.02rem; line-height: 1.15; }
    .brand-sub { font-size: 0.75rem; color: var(--muted) !important; }

    [data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: var(--wash);
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 14px;
        width: 100%;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: var(--teal-soft);
        border-color: #bcdadd;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--teal-soft);
        border-color: var(--teal);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: var(--teal-dark) !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

    /* ---------- Text ---------- */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; opacity: 1 !important; }
    h1, h2, h3, h4, h5, h6 {
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        font-weight: 800 !important;
    }
    h2 { font-size: 1.55rem !important; }
    h3 {
        font-size: 1.2rem !important;
        padding: 2px 0 2px 12px !important;
        line-height: 1.3 !important;
        border-left: 4px solid var(--teal);
        margin-top: 1.6rem !important;
    }
    a { color: var(--teal) !important; font-weight: 600; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #0f5c66 0%, #167d88 60%, #28a0aa 100%);
        border-radius: 18px;
        padding: 34px 38px;
        margin: 0.2rem 0 1.6rem;
        box-shadow: 0 14px 34px rgba(15, 92, 102, 0.25);
    }
    .hero::before, .hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
    }
    .hero::before { width: 260px; height: 260px; right: -60px; top: -90px; }
    .hero::after  { width: 160px; height: 160px; right: 150px; bottom: -90px; }
    .kicker {
        color: #bfe7ea !important;
        font-size: 0.74rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 2.3rem;
        margin: 0; line-height: 1.15;
    }
    .hero-sub {
        color: #def3f4 !important;
        margin: 0.7rem 0 0; font-size: 1.05rem; max-width: 620px;
    }
    .pills { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
    .pill {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #ffffff !important;
        padding: 5px 13px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }

    .lead, .muted { color: var(--muted) !important; }
    .lead { font-size: 1rem; margin: -0.3rem 0 1.1rem; }
    .muted { font-size: 0.9rem; }

    .notice {
        color: #29414d !important;
        background: #fff8e6;
        border: 1px solid #f1e0a8;
        border-left: 5px solid #e0a82e;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 1rem 0 1.4rem;
    }
    .notice strong { color: var(--ink) !important; }

    /* ---------- Cards ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(20, 48, 61, 0.06);
        padding: 6px 6px;
    }
    .card-title {
        display: flex; align-items: center; gap: 10px;
        color: var(--ink) !important;
        font-size: 1.05rem; font-weight: 700;
        margin: 0 0 0.6rem;
    }
    .step {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%;
        background: var(--teal); color: #fff !important;
        font-size: 0.8rem; font-weight: 700;
    }

    [data-testid="stImage"] img { border-radius: 12px; }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--line);
        border-top: 4px solid var(--teal);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(20, 48, 61, 0.05);
    }
    [data-testid="stMetricLabel"] * { color: var(--muted) !important; font-weight: 600; }
    [data-testid="stMetricValue"] * {
        color: var(--teal-dark) !important;
        font-weight: 800 !important;
    }

    /* ---------- Upload + controls ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--teal-soft) !important;
        border: 2px dashed #7fbcc2 !important;
        border-radius: 14px !important;
        padding: 22px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal) !important; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important;
        color: var(--teal-dark) !important;
        border: 1px solid #9fcfd3 !important;
        border-radius: 8px !important;
        font-weight: 600;
    }

    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #1b97a3, #0f5c66) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.3rem !important;
        font-weight: 650;
        box-shadow: 0 6px 14px rgba(22, 125, 136, 0.28);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 18px rgba(22, 125, 136, 0.34);
        color: #ffffff !important;
    }
    .stDownloadButton > button p { color: #ffffff !important; }

    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid #c7e1e1 !important;
        background: var(--teal-soft) !important;
    }
    [data-testid="stAlert"] * { color: var(--ink) !important; }

    input, textarea { background: #ffffff !important; color: var(--ink) !important; }
    hr { border-color: var(--line) !important; }
    footer { visibility: hidden; }

    /* ---------- Tables ---------- */
    .tbl-wrap {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(20, 48, 61, 0.05);
        margin-bottom: 0.6rem;
    }
    table.tbl { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
    table.tbl th {
        background: var(--teal-soft);
        color: var(--teal-dark) !important;
        text-align: left; font-weight: 700;
        padding: 12px 16px;
        border-bottom: 1px solid var(--line);
    }
    table.tbl td {
        color: var(--ink) !important;
        padding: 12px 16px;
        border-bottom: 1px solid #edf2f5;
    }
    table.tbl tr:last-child td { border-bottom: 0; }
    table.tbl tbody tr:hover td { background: #f7fbfb; }
    table.tbl td.best { color: var(--good) !important; font-weight: 700; }
    table.tbl td.first { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_table(rows, best_cols=()):
    """Show a list of dicts as a styled table (same data as st.dataframe)."""
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for row in rows:
        cells = ""
        for i, c in enumerate(cols):
            cls = "first" if i == 0 else ("best" if c in best_cols else "")
            value = f"{row[c]:.4f}" if isinstance(row[c], float) else row[c]
            cells += f'<td class="{cls}">{value}</td>'
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def header():
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Course project - Deep learning with computer vision</div>
          <h1>Brain Tumor Segmentation</h1>
          <div class="hero-sub">
            A 2D U-Net demonstration for segmenting tumor regions in brain MRI slices.
          </div>
          <div class="pills">
            <span class="pill">2D U-Net</span>
            <span class="pill">Figshare brain tumor dataset</span>
            <span class="pill">256 x 256 input</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preprocess(image: Image.Image):
    """Apply the same normalization and resize used during model training."""
    if image.mode in ("1", "L", "I", "I;16", "I;16B", "I;16L", "F"):
        array = np.asarray(image).astype(np.float32)
    else:
        array = np.asarray(image.convert("L")).astype(np.float32)

    if array.ndim != 2:
        raise ValueError("Please upload one two-dimensional MRI slice.")

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = array[array > 0]

    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = 0.0, 1.0

    scale = max(float(high - low), 1e-6)
    array = np.clip((array - float(low)) / scale, 0.0, 1.0)

    resized = Image.fromarray(array).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    tensor = torch.from_numpy(
        np.asarray(resized, dtype=np.float32).copy()
    ).unsqueeze(0).unsqueeze(0)

    return tensor, resized


@st.cache_resource(show_spinner="Loading the segmentation model...")
def load_model(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model = UNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def segmentation_page():
    header()

    st.markdown("## MRI segmentation")
    st.markdown(
        '<div class="lead">Upload one compatible MRI slice to view the predicted tumor mask and overlay.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><strong>Coursework demonstration only.</strong> '
        "This model is not a medical device and must not be used for diagnosis "
        "or treatment. It accepts one 2D image slice; full 3D scans and DICOM "
        "files are not supported.</div>",
        unsafe_allow_html=True,
    )

    upload_col, settings_col = st.columns([1.25, 0.85], gap="large")

    with upload_col:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span class="step">1</span>Select an MRI image</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "PNG, JPG, or TIFF image",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                help="A grayscale T1 contrast-enhanced MRI slice is most similar to the training data.",
            )
            st.markdown(
                '<div class="muted">The image is normalized and resized to '
                "256 x 256 pixels.</div>",
                unsafe_allow_html=True,
            )

    with settings_col:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span class="step">2</span>Model settings</div>',
                unsafe_allow_html=True,
            )
            st.write("**Model:** 2D U-Net")
            st.write("**Checkpoint:** 20-epoch experiment")
            threshold = st.slider(
                "Segmentation threshold",
                0.10,
                0.90,
                0.50,
                0.05,
                help="Project metrics use a threshold of 0.50.",
            )
            compute = "NVIDIA GPU" if torch.cuda.is_available() else "CPU"
            st.markdown(
                f'<div class="muted">Compute: {compute}</div>',
                unsafe_allow_html=True,
            )

    if uploaded is None:
        st.info("Select an image to see the MRI, predicted mask, and overlay.")
        return

    if not CHECKPOINT_PATH.exists():
        st.error(f"Model checkpoint not found: {CHECKPOINT_PATH}")
        return

    try:
        source = Image.open(io.BytesIO(uploaded.getvalue()))
        tensor, resized = preprocess(source)
    except Exception as exc:
        st.error(f"Could not read this as a compatible MRI slice: {exc}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = load_model(str(CHECKPOINT_PATH), device)
        with torch.no_grad():
            probability = torch.sigmoid(
                model(tensor.to(device))
            )[0, 0].cpu().numpy()
    except Exception as exc:
        st.error(f"Could not load the model or generate a prediction: {exc}")
        return

    mask = (probability >= threshold).astype(np.uint8)
    gray = np.clip(
        np.asarray(resized, dtype=np.float32) * 255,
        0,
        255,
    ).astype(np.uint8)

    overlay = np.stack([gray, gray, gray], axis=-1)
    selected = mask.astype(bool)
    overlay[selected] = (
        0.58 * overlay[selected]
        + 0.42 * np.array([200, 55, 65])
    ).astype(np.uint8)

    st.markdown("## Prediction results")
    st.markdown(
        '<div class="lead">All views use the same resized image coordinates.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Input slice</div>',
                unsafe_allow_html=True,
            )
            st.image(gray, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Predicted mask</div>',
                unsafe_allow_html=True,
            )
            st.image(mask * 255, clamp=True, use_container_width=True)

    with col3:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Tumor overlay</div>',
                unsafe_allow_html=True,
            )
            st.image(overlay, use_container_width=True)

    metric_col, note_col = st.columns([1, 2])

    with metric_col:
        st.metric("Area marked by the model", f"{mask.mean() * 100:.2f}%")

    with note_col:
        st.markdown(
            '<div class="muted" style="padding:1rem 0">'
            "This is the percentage of pixels marked in the resized slice, "
            "not a clinical measurement.</div>",
            unsafe_allow_html=True,
        )

    mask_bytes = io.BytesIO()
    Image.fromarray(mask * 255).save(mask_bytes, format="PNG")

    st.download_button(
        "Download predicted mask",
        mask_bytes.getvalue(),
        "predicted_tumor_mask.png",
        "image/png",
    )


def information_page():
    header()

    st.markdown("## Project overview")
    st.markdown(
        '<div class="lead">Dataset, model setup, experiment comparison, and held-out test results.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MRI slices", "3,064")
    k2.metric("Patients", "233")
    k3.metric("Test slices", "542")
    k4.metric("20-epoch test Dice", "0.7241")

    st.markdown("### Dataset")
    with st.container(border=True):
        st.markdown("**Figshare Brain Tumor dataset (Cheng et al.)**")
        st.write(
            "3,064 contrast-enhanced T1 MRI slices from 233 patients, with "
            "binary tumor masks and labels for glioma, meningioma, and pituitary "
            "tumors. This project uses 2,522 training slices and 542 held-out "
            "test slices."
        )
        st.markdown(
            "Source: [Figshare dataset](https://doi.org/10.6084/m9.figshare.1512427) "
            "- Cheng et al., *PLOS ONE* (2015), DOI: 10.1371/journal.pone.0140381."
        )

    st.markdown("### Method and training setup")
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Data preparation</div>',
                unsafe_allow_html=True,
            )
            st.write(
                "Patient-disjoint split: 2,041 training slices from 150 patients "
                "and 481 validation slices from 38 patients. The held-out test "
                "set has 542 slices from 45 patients. MRI intensities are "
                "normalized per slice; images and masks are resized to 256 x 256."
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Model and hardware</div>',
                unsafe_allow_html=True,
            )
            st.write(
                "2D U-Net with grayscale input and binary output. BCE + Dice "
                "loss, Adam optimizer, learning rate 0.0001, batch size 4. "
                "Trained on an NVIDIA RTX 3050 Ti Laptop GPU with 4 GB VRAM."
            )

    st.markdown("### Experiment comparison")
    html_table(
        [
            {
                "Experiment": "Baseline",
                "Epochs": 10,
                "Best validation Dice": 0.7203,
                "Test Dice": 0.6806,
                "Test IoU": 0.5950,
            },
            {
                "Experiment": "Extended training",
                "Epochs": 20,
                "Best validation Dice": 0.7640,
                "Test Dice": 0.7241,
                "Test IoU": 0.6280,
            },
        ],
        best_cols=("Epochs", "Best validation Dice", "Test Dice", "Test IoU"),
    )
    st.caption(
        "The extended run improved mean test Dice by 0.0435. Checkpoints: "
        "baseline_10_epochs.pt and unet_20_epochs.pt."
    )

    st.markdown("### Test scores by tumor type")
    html_table(
        [
            {
                "Tumor type": "Glioma",
                "Baseline Dice": 0.5576,
                "20-epoch Dice": 0.6150,
                "Baseline IoU": 0.4658,
                "20-epoch IoU": 0.5039,
                "Slices": 245,
            },
            {
                "Tumor type": "Meningioma",
                "Baseline Dice": 0.7722,
                "20-epoch Dice": 0.8408,
                "Baseline IoU": 0.7041,
                "20-epoch IoU": 0.7624,
                "Slices": 112,
            },
            {
                "Tumor type": "Pituitary",
                "Baseline Dice": 0.7880,
                "20-epoch Dice": 0.7980,
                "Baseline IoU": 0.7000,
                "20-epoch IoU": 0.7110,
                "Slices": 185,
            },
        ],
        best_cols=("20-epoch Dice", "20-epoch IoU"),
    )

    st.markdown("### Scope and limitations")
    st.info(
        "This is a coursework prototype for binary tumor-region segmentation "
        "in 2D slices, not a diagnostic tool. It supports PNG, JPG, and TIFF; "
        "it does not process full 3D NIfTI or DICOM studies."
    )
    st.caption(
        "Reference: Cheng, J. et al. (2015). Enhanced Performance of Brain "
        "Tumor Classification via Tumor Region Augmentation and Partition. "
        "PLOS ONE, 10(10), e0140381."
    )


def main():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-logo">🧠</div>
              <div>
                <div class="brand-name">Tumor Segmentation</div>
                <div class="brand-sub">Coursework demo</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            ["MRI Segmentation", "Project Information"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Student coursework demonstration**")
        st.caption("Not affiliated with a government agency or medical provider.")

    if page == "MRI Segmentation":
        segmentation_page()
    else:
        information_page()


if __name__ == "__main__":
    main()