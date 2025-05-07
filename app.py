import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# 📌 ชื่อไฟล์ weight ที่คุณอัปโหลดไว้ใน GitHub Repo
MODEL_PATH = 'my_checkpoint.weights.h5'
IMG_SIZE = (224, 224)

# 🧠 โมเดลต้องใช้ EfficientNetB3 แบบเดียวกับตอนเทรนใน Colab
def build_model():
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    base_model.trainable = False  # Freeze base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(5, activation='softmax')  # 🔁 แก้เป็นจำนวน class จริงถ้ามีมากกว่า 5
    ])
    return model

# ✅ โหลด weights ที่บันทึกไว้จาก Colab
@st.cache_resource(show_spinner='Loading model…')
def get_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

# ✅ ถ้าไม่มี labels.txt หรือใน .h5 ไม่มี class name ให้กำหนดเอง
@st.cache_resource
def load_class_names() -> list[str]:
    return [f'class_{i}' for i in range(5)]  # 🔁 ใส่ชื่อจริง เช่น ['cat', 'dog', 'horse', ...]

CLASS_NAMES = load_class_names()
model = get_model()

# 📸 เตรียม UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to classify it.")

uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# 🔁 เตรียมภาพก่อนทำนาย
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# 🔮 ทำนายภาพ
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
