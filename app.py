import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# 📌 ชื่อไฟล์ weights จาก Colab
MODEL_PATH = 'my_checkpoint.weights.h5'
IMG_SIZE = (224, 224)

# ✅ ใช้ EfficientNetB3 แบบที่คุณเทรนไว้ใน Colab
def build_model():
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    base_model.trainable = False

    # ✅ ต่อด้วย layer ที่คุณใช้ตอน train
    x = tf.keras.layers.Dense(5, activation='softmax')(base_model.output)  # 🔁 เปลี่ยน 5 เป็นจำนวนคลาสจริง
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

# ✅ โหลด weights
@st.cache_resource(show_spinner='Loading model…')
def get_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

@st.cache_resource
def load_class_names() -> list[str]:
    return [f'class_{i}' for i in range(5)]  # เปลี่ยนชื่อ class ตามจริงถ้ามี

model = get_model()
CLASS_NAMES = load_class_names()

# 🔁 เตรียม UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Classifier Demo")
st.write("Upload an image of an animal and click **Predict** to classify it.")

uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

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
