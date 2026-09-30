# Hindi Character Recognition — Drawing App

This version removes image upload. The user draws one Hindi character directly on a canvas.

The app simultaneously shows:
- Perceptron prediction
- ANN prediction
- CNN prediction
- confidence for each model
- model agreement/disagreement
- top 5 CNN predictions
- the processed 28x28 input

## Project structure

```text
hindi-character-recognition/
├── app.py
├── requirements.txt
└── models/
    ├── perceptron.keras
    ├── ann.keras
    └── cnn.keras
```

## Requirements

`streamlit-drawable-canvas` provides the freehand drawing canvas in Streamlit. Version 0.13.0 was released in September 2026 and supports Python 3.10–3.13. 

## Deployment

Use Python 3.12 or 3.13 on Streamlit Community Cloud.

Commit all files to GitHub and deploy `app.py`.

## Model input assumptions

This app assumes:
- Perceptron and ANN accept `(28, 28)`
- CNN accepts `(28, 28, 1)`
- black background / white character
- pixel values normalized to 0–1
- 50 output classes

The numeric labels are converted to the Hindi characters using the same class order used during training.
