import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

# 📌 ตั้งชื่อไฟล์ให้ตรงกับที่คุณ push ขึ้น GitHub
MODEL_PATH = 'my_checkpoint.weights.h5'
IMG_SIZE = (224, 224)

# 🧠 สร้างโครงสร้างโมเดลให้ตรงกับของจริงที่คุณใช้ตอนเทรน
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax')  # ✅ เปลี่ยน 5 เป็นจำนวน class จริงของคุณ
    ])
    return model

# ✅ โหลด weights แทน load_model เพราะไม่มี config ใน .weights.h5
@st.cache_resource(show_spinner='Loading model…')
def get_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

# ✅ ตั้งชื่อคลาสเอง ถ้าไม่มี class_names ในไฟล์
@st.cache_resource
def load_class_names() -> list[str]:
    return [f'class_{i}' for i in range(5)]  # ✅ แก้เป็นชื่อจริงถ้ามี เช่น ['cat', 'dog', ...]

CLASS_NAMES = load_class_names()
model = get_model()

# ✨ เตรียม UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to classify it.")

# 📸 โหลดภาพจากผู้ใช้
uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# 🧪 เตรียมรูปภาพก่อนใช้โมเดล
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# 🔍 ทำการพยากรณ์
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict(x, verbose=0)[0]     # shape = (n_classes,)
        top_k = preds.argsort()[-5:][::-1]          # top‑5

        st.subheader("Prediction (Top‑5)")
        for i in top_k:
            st.write(f"- **{CLASS_NAMES[i]}** : {preds[i]*100:.2f}%")
