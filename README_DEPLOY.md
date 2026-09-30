# Hindi Character Recognition Deployment

This deployment runs the three Keras models from the Hindi character notebook:

- `perceptron.keras`
- `ann.keras`
- `cnn.keras`

The Streamlit app accepts a single Hindi character image and displays predictions from all three models.

## 1. Save the models in Colab

Run this cell **after all three models have finished training**:

```python
import os
os.makedirs("/content/models", exist_ok=True)

perceptron.save("/content/models/perceptron.keras")
ann.save("/content/models/ann.keras")
cnn.save("/content/models/cnn.keras")
```

Your notebook currently has the save cell before the CNN definition, so make sure you execute saving only after `cnn.fit(...)` has completed.

Then download the three `.keras` files.

## 2. Project structure

```text
hindi-character-recognition/
│
├── app.py
├── requirements.txt
│
└── models/
    ├── perceptron.keras
    ├── ann.keras
    └── cnn.keras
```

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy with Streamlit Community Cloud

Push this project to GitHub, including `app.py`, `requirements.txt`, and the three model files under `models/`.

Then open Streamlit Community Cloud, choose the GitHub repository and select `app.py` as the entrypoint.

## Important

The preprocessing in the app is intended to approximate the 28x28 grayscale format used by the training notebook. For highest real-world accuracy, training data and deployment preprocessing should be made as similar as possible.

The model labels remain numeric internally (0-49); the app converts them to the Hindi character names using the same 50-class mapping used by the notebook.
