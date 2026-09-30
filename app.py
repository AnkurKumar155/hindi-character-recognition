import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_models():
    perceptron = tf.keras.models.load_model(MODEL_DIR / "perceptron.keras")
    ann = tf.keras.models.load_model(MODEL_DIR / "ann.keras")
    cnn = tf.keras.models.load_model(MODEL_DIR / "cnn.keras")
    return perceptron, ann, cnn

def preprocess_image(uploaded_file):
    """
    Convert a user image into the same basic format used during training:
    grayscale -> foreground/background normalization -> square -> 28x28 -> [0,1].
    """
    img = Image.open(uploaded_file).convert("L")
    arr = np.array(img)

    # Make dark background / bright character, matching the training images.
    # If the image is mostly white, invert it.
    if arr.mean() > 127:
        arr = 255 - arr

    # Remove very weak background noise.
    arr[arr < 25] = 0

    ys, xs = np.where(arr > 25)

    if len(xs) > 0:
        # Crop to foreground with a small margin.
        x0, x1 = max(0, xs.min() - 5), min(arr.shape[1], xs.max() + 6)
        y0, y1 = max(0, ys.min() - 5), min(arr.shape[0], ys.max() + 6)
        arr = arr[y0:y1, x0:x1]

    img = Image.fromarray(arr.astype(np.uint8))

    # Pad to a square.
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))

    # Keep the character away from the borders.
    target = 22
    if side > 0:
        scale = target / side
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(
            resized,
            ((28 - nw) // 2, (28 - nh) // 2)
        )
    else:
        canvas = Image.new("L", (28, 28), 0)

    # Match model input range.
    x = np.array(canvas, dtype=np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)

    return x

def predict_all(x28):
    # Perceptron and ANN were trained with (28, 28).
    x_ann = x28[np.newaxis, ...]

    # CNN was trained with (28, 28, 1).
    x_cnn = x28[np.newaxis, ..., np.newaxis]

    perceptron, ann, cnn = load_models()

    per_prob = perceptron.predict(x_ann, verbose=0)[0]
    ann_prob = ann.predict(x_ann, verbose=0)[0]
    cnn_prob = cnn.predict(x_cnn, verbose=0)[0]

    return per_prob, ann_prob, cnn_prob

st.title("Hindi Character Recognition")
st.write("Compare predictions from Perceptron, ANN and CNN.")

with st.sidebar:
    st.header("Model")
    selected_model = st.selectbox(
        "Main model to display",
        ["CNN", "ANN", "Perceptron"]
    )
    st.write("50 Hindi character classes")

uploaded = st.file_uploader(
    "Upload a Hindi character image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is None:
    st.info("Upload one character image to get predictions.")
    st.stop()

x28 = preprocess_image(uploaded)

per_prob, ann_prob, cnn_prob = predict_all(x28)

prob_map = {
    "Perceptron": per_prob,
    "ANN": ann_prob,
    "CNN": cnn_prob
}

# Main prediction
main_prob = prob_map[selected_model]
main_label = int(np.argmax(main_prob))
main_conf = float(main_prob[main_label]) * 100

col1, col2 = st.columns(2)

with col1:
    st.subheader("Processed image")
    st.image(x28, width=220, clamp=True)

with col2:
    st.subheader(f"{selected_model} prediction")
    st.metric(
        label="Predicted character",
        value=HINDI_LABELS[main_label],
        delta=f"{main_conf:.2f}% confidence"
    )
    st.caption(f"Class label: {main_label}")

st.divider()
st.subheader("All model predictions")

results = []
for name in ["Perceptron", "ANN", "CNN"]:
    prob = prob_map[name]
    label = int(np.argmax(prob))
    conf = float(prob[label]) * 100
    results.append((name, HINDI_LABELS[label], label, conf))

cols = st.columns(3)
for col, (name, char, label, conf) in zip(cols, results):
    with col:
        st.markdown(f"### {name}")
        st.markdown(f"# {char}")
        st.write(f"Class: **{label}**")
        st.write(f"Confidence: **{conf:.2f}%**")

st.divider()

st.subheader("Top 5 CNN predictions")
top5 = np.argsort(cnn_prob)[::-1][:5]
for rank, label in enumerate(top5, start=1):
    st.write(
        f"{rank}. **{HINDI_LABELS[int(label)]}** "
        f"(class {int(label)}) — {float(cnn_prob[label])*100:.2f}%"
    )
