import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from streamlit_drawable_canvas import st_canvas


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon="अ",
    layout="wide"
)


# =========================================================
# HINDI LABELS
# Must be exactly the same order used during training
# =========================================================

HINDI_LABELS = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "अं", "अः",
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ", "श्र"
]


# =========================================================
# MODEL PATH
# Your GitHub screenshot showed the models are in the
# same directory as app.py
#
# app.py
# perceptron.keras
# ann.keras
# cnn.keras
# =========================================================

MODEL_DIR = Path(__file__).resolve().parent


# =========================================================
# LOAD MODELS
# compile=False avoids loading old optimizer/loss state
# =========================================================

@st.cache_resource
def load_models():

    perceptron_path = MODEL_DIR / "perceptron.keras"
    ann_path = MODEL_DIR / "ann.keras"
    cnn_path = MODEL_DIR / "cnn.keras"

    # Check model files
    if not perceptron_path.exists():
        st.error(f"Perceptron model not found: {perceptron_path}")
        st.stop()

    if not ann_path.exists():
        st.error(f"ANN model not found: {ann_path}")
        st.stop()

    if not cnn_path.exists():
        st.error(f"CNN model not found: {cnn_path}")
        st.stop()

    try:

        perceptron = tf.keras.models.load_model(
            perceptron_path,
            compile=False
        )

        ann = tf.keras.models.load_model(
            ann_path,
            compile=False
        )

        cnn = tf.keras.models.load_model(
            cnn_path,
            compile=False
        )

        return perceptron, ann, cnn

    except Exception as e:
        st.error("Model loading failed")
        st.exception(e)
        st.stop()


perceptron, ann, cnn = load_models()


# =========================================================
# PREPROCESS DRAWING
#
# Canvas:
# 400 x 400
#
# Model:
# 28 x 28 grayscale
#
# Training images:
# black background
# white character
# pixel range 0-1
# =========================================================

def preprocess_canvas(image_data):

    # image_data returned by canvas is RGBA
    if image_data.shape[-1] == 4:

        # Ignore alpha channel
        gray = image_data[:, :, :3].mean(axis=2)

    else:

        gray = image_data.mean(axis=2)

    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # -----------------------------------------------------
    # Find character pixels
    # -----------------------------------------------------

    ys, xs = np.where(gray > 25)

    # Empty canvas
    if len(xs) == 0:
        return None

    # -----------------------------------------------------
    # Crop around character
    # -----------------------------------------------------

    x0 = max(0, xs.min() - 15)
    x1 = min(gray.shape[1], xs.max() + 16)

    y0 = max(0, ys.min() - 15)
    y1 = min(gray.shape[0], ys.max() + 16)

    cropped = Image.fromarray(
        gray[y0:y1, x0:x1],
        mode="L"
    )

    # -----------------------------------------------------
    # Make square
    # -----------------------------------------------------

    w, h = cropped.size

    side = max(w, h)

    square = Image.new(
        "L",
        (side, side),
        0
    )

    square.paste(
        cropped,
        ((side - w) // 2,
         (side - h) // 2)
    )

    # -----------------------------------------------------
    # Resize character to approximately MNIST size
    # -----------------------------------------------------

    target_size = 22

    scale = target_size / side

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cropped.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # -----------------------------------------------------
    # Final 28x28 image
    # -----------------------------------------------------

    final = Image.new(
        "L",
        (28, 28),
        0
    )

    final.paste(
        resized,
        (
            (28 - new_w) // 2,
            (28 - new_h) // 2
        )
    )

    # -----------------------------------------------------
    # Convert to numpy
    # -----------------------------------------------------

    x = np.asarray(
        final,
        dtype=np.float32
    )

    # Normalize exactly like training
    x = x / 255.0

    x = np.clip(
        x,
        0.0,
        1.0
    )

    return x


# =========================================================
# PREDICTION
#
# Perceptron input = (1,28,28)
# ANN input        = (1,28,28)
# CNN input        = (1,28,28,1)
# =========================================================

def predict_models(x28):

    # -----------------------------------------------------
    # Perceptron and ANN
    # Your notebook trained:
    #
    # Flatten(input_shape=(28,28))
    #
    # Therefore input remains (1,28,28)
    # -----------------------------------------------------

    x_dense = x28[np.newaxis, ...]


    # -----------------------------------------------------
    # CNN
    # Your CNN trained with:
    #
    # Input(shape=(28,28,1))
    #
    # -----------------------------------------------------

    x_cnn = x28[np.newaxis, ..., np.newaxis]


    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    per_prob = perceptron.predict(
        x_dense,
        verbose=0
    )[0]

    ann_prob = ann.predict(
        x_dense,
        verbose=0
    )[0]

    cnn_prob = cnn.predict(
        x_cnn,
        verbose=0
    )[0]


    # -----------------------------------------------------
    # Get class index
    # -----------------------------------------------------

    per_label = int(
        np.argmax(per_prob)
    )

    ann_label = int(
        np.argmax(ann_prob)
    )

    cnn_label = int(
        np.argmax(cnn_prob)
    )


    # -----------------------------------------------------
    # Return Hindi characters + confidence
    # -----------------------------------------------------

    return {

        "perceptron": {
            "label": per_label,
            "character": HINDI_LABELS[per_label],
            "confidence": float(
                per_prob[per_label]
            ) * 100
        },

        "ann": {
            "label": ann_label,
            "character": HINDI_LABELS[ann_label],
            "confidence": float(
                ann_prob[ann_label]
            ) * 100
        },

        "cnn": {
            "label": cnn_label,
            "character": HINDI_LABELS[cnn_label],
            "confidence": float(
                cnn_prob[cnn_label]
            ) * 100
        },

        "cnn_prob": cnn_prob
    }


# =========================================================
# TITLE
# =========================================================

st.title("🇮🇳 Hindi Character Recognition")

st.write(
    "Draw one Hindi character below. "
    "Perceptron, ANN and CNN predictions "
    "will be displayed together."
)


# =========================================================
# MAIN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [1, 1.35]
)


# =========================================================
# DRAWING AREA
# =========================================================

with left_col:

    st.subheader("Draw Character")

    # Brush size
    stroke_width = st.slider(
        "Brush size",
        min_value=5,
        max_value=40,
        value=18
    )

    st.caption(
        "Write one Hindi character inside the black box."
    )


    # -----------------------------------------------------
    # IMPORTANT:
    # display_toolbar=True removed
    # because it is not part of the current 0.13 API
    #
    # return_image_data=True is required
    # -----------------------------------------------------

    canvas_result = st_canvas(

        fill_color="rgba(0, 0, 0, 0)",

        stroke_width=stroke_width,

        stroke_color="#FFFFFF",

        background_color="#000000",

        height=400,

        width=400,

        drawing_mode="freedraw",

        update_streamlit=True,

        return_image_data=True,

        key="hindi_canvas"
    )


# =========================================================
# PROCESS DRAWING
# =========================================================

processed = None

if canvas_result.image_data is not None:

    processed = preprocess_canvas(
        canvas_result.image_data
    )


# =========================================================
# PREDICTIONS
# =========================================================

with right_col:

    st.subheader("Predictions")


    # Empty canvas
    if processed is None:

        st.info(
            "Draw a Hindi character to see "
            "all three model predictions."
        )


    else:

        results = predict_models(
            processed
        )


        # -------------------------------------------------
        # THREE MODEL RESULTS
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)


        # -------------------------------------------------
        # PERCEPTRON
        # -------------------------------------------------

        with col1:

            st.markdown("### Perceptron")

            st.markdown(
                f"# {results['perceptron']['character']}"
            )

            st.write(
                f"Confidence: "
                f"**{results['perceptron']['confidence']:.2f}%**"
            )

            st.write(
                f"Class: "
                f"**{results['perceptron']['label']}**"
            )


        # -------------------------------------------------
        # ANN
        # -------------------------------------------------

        with col2:

            st.markdown("### ANN")

            st.markdown(
                f"# {results['ann']['character']}"
            )

            st.write(
                f"Confidence: "
                f"**{results['ann']['confidence']:.2f}%**"
            )

            st.write(
                f"Class: "
                f"**{results['ann']['label']}**"
            )


        # -------------------------------------------------
        # CNN
        # -------------------------------------------------

        with col3:

            st.markdown("### CNN")

            st.markdown(
                f"# {results['cnn']['character']}"
            )

            st.write(
                f"Confidence: "
                f"**{results['cnn']['confidence']:.2f}%**"
            )

            st.write(
                f"Class: "
                f"**{results['cnn']['label']}**"
            )


        # =================================================
        # MODEL AGREEMENT
        # =================================================

        st.divider()

        st.subheader(
            "Model Agreement"
        )

        per_char = results[
            "perceptron"
        ]["character"]

        ann_char = results[
            "ann"
        ]["character"]

        cnn_char = results[
            "cnn"
        ]["character"]


        if (
            per_char == ann_char
            and
            ann_char == cnn_char
        ):

            st.success(
                f"✅ All three models predict: "
                f"**{cnn_char}**"
            )

        else:

            st.warning(
                f"Predictions differ — "
                f"Perceptron: **{per_char}**, "
                f"ANN: **{ann_char}**, "
                f"CNN: **{cnn_char}**"
            )


        # =================================================
        # PROCESSED IMAGE
        # =================================================

        st.subheader(
            "Processed 28 × 28 Input"
        )

        st.image(
            processed,
            width=180,
            clamp=True
        )


        # =================================================
        # TOP 5 CNN PREDICTIONS
        # =================================================

        st.subheader(
            "Top 5 CNN Predictions"
        )

        cnn_prob = results[
            "cnn_prob"
        ]

        top5 = np.argsort(
            cnn_prob
        )[::-1][:5]


        for rank, label in enumerate(
            top5,
            start=1
        ):

            label = int(label)

            confidence = (
                float(
                    cnn_prob[label]
                ) * 100
            )

            st.write(
                f"{rank}. "
                f"**{HINDI_LABELS[label]}** "
                f"(Class {label}) — "
                f"**{confidence:.2f}%**"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "50 Hindi character classes • "
    "28×28 grayscale • "
    "Perceptron + ANN + CNN"
)
