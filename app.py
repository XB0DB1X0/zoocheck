import streamlit as st
from PIL import Image
import numpy as np
import joblib  # <– ใช้แทน tf.keras
from pathlib import Path

MODEL_PATH = 'my_checkpoint.pkl'

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_class_names() -> list[str]:
    return [f'class_{i}' for i in range(5)]  # หรือ list ชื่อจริง

CLASS_NAMES = load_class_names()
IMG_SIZE = (224, 224)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return arr.flatten().reshape(1, -1)  # แบบ flatten สำหรับ sklearn

model = get_model()

st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict**")

uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict_proba(x)[0]  # ต้องใช้ predict_proba สำหรับ Top-5
        top_k = preds.argsort()[-5:][::-1]

        st.subheader("Prediction (Top‑5)")
        for i in top_k:
            st.write(f"- **{CLASS_NAMES[i]}** : {preds[i]*100:.2f}%")
