import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon="अ",
    layout="wide"
)

HINDI_LABELS = [
    "अ","आ","इ","ई","उ","ऊ","ऋ","ए","ऐ","ओ","औ","अं","अः",
    "क","ख","ग","घ","ङ","च","छ","ज","झ","ञ",
    "ट","ठ","ड","ढ","ण",
    "त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व",
    "श","ष","स","ह","क्ष","त्र","ज्ञ","श्र"
]

MODEL_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Load models once
# ---------------------------------------------------------
#@st.cache_resource
@st.cache_resource
def load_models():

    model_files = {
        "Perceptron": MODEL_DIR / "perceptron.keras",
        "ANN": MODEL_DIR / "ann.keras",
        "CNN": MODEL_DIR / "cnn.keras"
    }

    # Check files first
    for name, path in model_files.items():
        if not path.exists():
            st.error(f"{name} model not found: {path}")
            st.stop()

        st.write(
            f"{name}: {path.name} | "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

    try:
        perceptron = tf.keras.models.load_model(
            model_files["Perceptron"],
            compile=False
        )

        ann = tf.keras.models.load_model(
            model_files["ANN"],
            compile=False
        )

        cnn = tf.keras.models.load_model(
            model_files["CNN"],
            compile=False
        )

        return perceptron, ann, cnn

    except Exception as e:
        st.error("Model loading failed")
        st.exception(e)
        st.stop()

perceptron, ann, cnn = load_models()

# ---------------------------------------------------------
# Convert canvas to 28x28
# Training data: black background + white character
# ---------------------------------------------------------
def preprocess_canvas(image_data):
    # RGBA/RGB -> grayscale
    if image_data.shape[-1] == 4:
        gray = image_data[:, :, :3].mean(axis=2)
    else:
        gray = image_data.mean(axis=2)

    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Find character pixels
    ys, xs = np.where(gray > 25)

    if len(xs) == 0:
        return None

    # Crop character
    x0 = max(0, xs.min() - 15)
    x1 = min(gray.shape[1], xs.max() + 16)
    y0 = max(0, ys.min() - 15)
    y1 = min(gray.shape[0], ys.max() + 16)

    cropped = Image.fromarray(gray[y0:y1, x0:x1], mode="L")

    # Resize while preserving aspect ratio
    w, h = cropped.size
    side = max(w, h)

    square = Image.new("L", (side, side), 0)
    square.paste(
        cropped,
        ((side - w) // 2, (side - h) // 2)
    )

    # Keep a little margin like MNIST
    target_size = 22
    scale = target_size / side

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)

    final = Image.new("L", (28, 28), 0)
    final.paste(
        resized,
        ((28 - nw) // 2, (28 - nh) // 2)
    )

    x = np.asarray(final, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_models(x28):
    # Perceptron / ANN
    x_dense = x28[np.newaxis, ...]

    # CNN
    x_cnn = x28[np.newaxis, ..., np.newaxis]

    per_prob = perceptron.predict(x_dense, verbose=0)[0]
    ann_prob = ann.predict(x_dense, verbose=0)[0]
    cnn_prob = cnn.predict(x_cnn, verbose=0)[0]

    per_label = int(np.argmax(per_prob))
    ann_label = int(np.argmax(ann_prob))
    cnn_label = int(np.argmax(cnn_prob))

    return (
        HINDI_LABELS[per_label], float(per_prob[per_label]) * 100,
        HINDI_LABELS[ann_label], float(ann_prob[ann_label]) * 100,
        HINDI_LABELS[cnn_label], float(cnn_prob[cnn_label]) * 100,
        cnn_prob
    )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🇮🇳 Hindi Character Recognition")
st.caption("Draw one Hindi character. Perceptron, ANN and CNN predictions are shown together.")

left, right = st.columns([1, 1.35])

with left:
    st.subheader("Draw Character")

    stroke_width = st.slider(
        "Brush size",
        min_value=5,
        max_value=40,
        value=18
    )

    canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=400,
    width=400,
    drawing_mode="freedraw",
    update_streamlit=True,
    display_toolbar=True,
    return_image_data=True,
    key="hindi_canvas"
)

if canvas_result.image_data is not None:
    processed = preprocess_canvas(canvas_result.image_data)

    st.caption("Write a single character in the black box.")

with right:
    st.subheader("Predictions")

    processed = None
    if canvas_result.image_data is not None:
        processed = preprocess_canvas(canvas_result.image_data)

    if processed is None:
        st.info("Draw a Hindi character to see all three model predictions.")
    else:
        # Automatically predict after drawing.
        (
            per_char, per_conf,
            ann_char, ann_conf,
            cnn_char, cnn_conf,
            cnn_prob
        ) = predict_models(processed)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Perceptron")
            st.markdown(f"# {per_char}")
            st.write(f"Confidence: **{per_conf:.2f}%**")

        with col2:
            st.markdown("### ANN")
            st.markdown(f"# {ann_char}")
            st.write(f"Confidence: **{ann_conf:.2f}%**")

        with col3:
            st.markdown("### CNN")
            st.markdown(f"# {cnn_char}")
            st.write(f"Confidence: **{cnn_conf:.2f}%**")

        st.divider()
        st.subheader("Model Agreement")

        if per_char == ann_char == cnn_char:
            st.success(f"All three models predict: **{cnn_char}**")
        else:
            st.warning(
                f"Predictions differ — Perceptron: **{per_char}**, "
                f"ANN: **{ann_char}**, CNN: **{cnn_char}**"
            )

        st.subheader("Processed 28×28 input")
        st.image(processed, width=180, clamp=True)

        st.subheader("Top 5 CNN predictions")
        top5 = np.argsort(cnn_prob)[::-1][:5]
        for rank, label in enumerate(top5, 1):
            st.write(
                f"{rank}. **{HINDI_LABELS[int(label)]}** — "
                f"{float(cnn_prob[int(label)]) * 100:.2f}%"
            )

st.divider()
st.caption("50 classes • 28×28 grayscale input • Perceptron + ANN + CNN")
