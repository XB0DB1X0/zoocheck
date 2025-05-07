import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

# 📌 ตั้งชื่อไฟล์ให้ตรงกับที่คุณ push ขึ้น GitHub
MODEL_PATH = 'my_checkpoint.weights.h5'
IMG_SIZE = (224, 224)

# 🧠 สร้างโครงสร้างโมเดลให้ตรงกับของจริงที่ใช้ใน Google Colab
def build_model():
    model = tf.keras.Sequential(name='animal_model')
    model.add(tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer'))
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv2d'))
    model.add(tf.keras.layers.MaxPooling2D(name='maxpool'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(5, activation='softmax', name='dense'))  # 🔁 แก้ 5 เป็นจำนวนคลาสจริง
    return model

# ✅ โหลด weights แบบถูกต้อง
@st.cache_resource(show_spinner='Loading model…')
def get_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

# ✅ ตั้งชื่อ class แบบกำหนดเอง
@st.cache_resource
def load_class_names() -> list[str]:
    return [f'class_{i}' for i in range(5)]  # 🔁 แก้ชื่อคลาสจริงถ้ามี เช่น ['cat', 'dog', ...]

CLASS_NAMES = load_class_names()
model = get_model()

# 🖼️ UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to classify it.")

uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# 🔁 เตรียมภาพ
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# 🔮 ทำนาย
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict(x, verbose=0)[0]
        top_k = preds.argsort()[-5:][::-1]

        st.subheader("Prediction (Top‑5)")
        for i in top_k:
            st.write(f"- **{CLASS_NAMES[i]}** : {preds[i]*100:.2f}%")
