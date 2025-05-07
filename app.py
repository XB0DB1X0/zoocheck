# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:17:30 2025
@author: hahah
"""
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
import h5py, json

# --------------------------------------------------------------------
# 1) Model path
MODEL_PATH = 'my_checkpoint.weights.h5'   # or a SavedModel directory

# --------------------------------------------------------------------
# 2) Load class names (from attr / assets / fallback)
@st.cache_resource
def load_class_names() -> list[str]:
    # --- Case 1: attribute inside a .h5 file --------------------------------
    if MODEL_PATH.endswith('.h5'):
        try:
            with h5py.File(MODEL_PATH, 'r') as f:
                raw = f.attrs.get('class_names')
                if raw is not None:
                    raw = raw.decode() if isinstance(raw, bytes) else raw
                    return json.loads(raw)
        except Exception:
            pass

    # --- Case 2: labels.txt inside assets/ (SavedModel) ---------------------
    p = Path(MODEL_PATH) / 'assets' / 'labels.txt'
    if p.exists():
        return [l.strip() for l in p.read_text(encoding='utf-8').splitlines()
                if l.strip()]

    # --- Case 3: fallback – create generic names ----------------------------
    tmp = tf.keras.models.load_model(MODEL_PATH)
    return [f'class_{i}' for i in range(tmp.output_shape[-1])]

CLASS_NAMES = load_class_names()
IMG_SIZE = (224, 224)                     # adjust to your model

# --------------------------------------------------------------------
# 3) Load model once (cached)
@st.cache_resource(show_spinner='Loading model…')
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --------------------------------------------------------------------
# 4) Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr   = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)     # shape = (1, H, W, 3)

# --------------------------------------------------------------------
# 5) Streamlit UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to let the "
         "model identify the species.")

uploaded = st.file_uploader(
    "Choose a .jpg / .jpeg / .png image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict(x, verbose=0)[0]          # shape = (n_classes,)
        top_k = preds.argsort()[-5:][::-1]              # Top‑5

        st.subheader("Prediction (Top‑5)")
        for i in top_k:
            st.write(f"- **{CLASS_NAMES[i]}** : {preds[i]*100:.2f}%")

        with st.expander(f"Show probabilities for all {len(CLASS_NAMES)} "
                         "species"):
            for i, p in enumerate(preds):
                st.write(f"{CLASS_NAMES[i]:>20} → {p*100:.2f}%")
