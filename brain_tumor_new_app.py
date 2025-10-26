import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Tumor Diagnosis", layout="wide", page_icon="🧠")

st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Diagnose MRI", "📊 About Project", "👩‍💻 Team"])

# -----------------------------
# LOAD MODEL
# -----------------------------


@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_model.h5", compile=False)
    dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
    model.predict(dummy_input)  # Build the model once
    return model


model = load_brain_model()
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']



# -----------------------------
# FUNCTION: Grad-CAM
# -----------------------------


# def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     return heatmap.numpy()



def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    # Ensure model is built
    _ = model.predict(img_array)

    # Automatically detect the last conv layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = [
            layer.name for layer in model.layers if "conv" in layer.name
        ][-1]

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -----------------------------
# Grad-CAM Visualization
# -----------------------------
try:
    heatmap = get_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (150,150))

    # Read image from upload directly
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, 1)
    uploaded_file.seek(0)

    img_orig = cv2.resize(img_orig, (150,150))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

    with col2:
        st.subheader("Grad-CAM Visualization")
        st.image(overlay, width='stretch', caption="Explainable AI Highlighted Regions")
except Exception as e:
    st.error(f"Grad-CAM failed: {e}")



# -----------------------------
# HOME TAB
# -----------------------------
if section == "🏠 Home":
    st.title("🧠 Brain Tumor Diagnosis using Explainable AI")
    st.markdown("""
    This application helps **diagnose brain tumors from MRI scans** using a **Convolutional Neural Network (CNN)** 
    combined with **Grad-CAM visual explanations**.  
    Upload an MRI image to view:
    - Model prediction & confidence
    - Explainable heatmap (Grad-CAM)
    - Downloadable diagnosis report
    """)

    #st.image("assets/banner.png", use_container_width=True)

   # st.image("https://cdn.pixabay.com/photo/2019/02/17/19/13/brain-4008705_960_720.jpg", use_container_width=True)

    #st.image("https://raw.githubusercontent.com/Thanvitha-mitta/brain-tumor/Brain_tumor_image.jpg", use_container_width=True)

    st.image("Brain_tumor_image.jpg", width='stretch')




# -----------------------------
# DIAGNOSE TAB
# -----------------------------
elif section == "🔍 Diagnose MRI":
    st.title("🔍 Brain Tumor Diagnosis")

    uploaded_file = st.file_uploader("Upload MRI Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # Display uploaded image
        with col1:
            st.subheader("Original MRI Image")
            st.image(uploaded_file, use_container_width=True)

        # Preprocess image
        img = image.load_img(uploaded_file, target_size=(150,150))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner("Analyzing MRI scan..."):
            time.sleep(2)
            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = class_labels[np.argmax(preds)]

        # Progress bar & result
        st.success(f"✅ Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
        st.progress(float(confidence))

        # -----------------------------
        # EXPLAINABLE CONFIDENCE INTERPRETATION
        # -----------------------------
        if confidence > 0.85:
            explain_text = "The model is **highly confident** in its prediction."
        elif 0.60 <= confidence <= 0.85:
            explain_text = "The model is **moderately confident**. Consider cross-verifying with other scans."
        else:
            explain_text = "The model has **low confidence** — tumor may not be clearly visible or image quality may be low."

        st.markdown(f"🧾 **Confidence Interpretation:** {explain_text}")

        # -----------------------------
        # Confidence Bar Chart
        # -----------------------------
        fig = go.Figure([go.Bar(
            x=class_labels,
            y=preds[0]*100,
            marker_color=['#4CAF50' if x == np.argmax(preds) else '#B0BEC5' for x in range(len(class_labels))]
        )])
        fig.update_layout(title="Prediction Confidence (%)", yaxis_title="Confidence (%)")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Grad-CAM Visualization
        # -----------------------------
        heatmap = get_gradcam_heatmap(img_array, model)
        heatmap = cv2.resize(heatmap, (150,150))
        # Convert uploaded file to NumPy for OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_orig = cv2.imdecode(file_bytes, 1)
        uploaded_file.seek(0)  # Reset pointer for other uses

        #img_orig = cv2.imread(uploaded_file.name)
        img_orig = cv2.resize(img_orig, (150,150))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

        with col2:
            st.subheader("Grad-CAM Visualization")
            st.image(overlay, use_container_width=True, caption="Explainable AI Highlighted Regions")

# -----------------------------
# ABOUT PROJECT TAB
# -----------------------------
elif section == "📊 About Project":
    st.header("📊 Project Overview")
    st.markdown("""
    - **Title:** An Application to Diagnose Brain Tumor using Explainable AI-Based Models  
    - **Model Used:** Convolutional Neural Network (CNN)  
    - **Explainability:** Grad-CAM heatmap for region-based interpretation  
    - **Dataset:** MRI Brain Scans (4 classes — Glioma, Meningioma, Pituitary, No Tumor)  
    - **Tools Used:** TensorFlow, OpenCV, Streamlit, Matplotlib, Plotly
    """)

# -----------------------------
# TEAM TAB
# -----------------------------
elif section == "👩‍💻 Team":
    st.header("👩‍💻 Project Team")
    st.markdown("""
    **Developed by:** Thanvitha Mitta  
    **Institution:** Bangalore Institute of Technology  
    **Guide:** [Your Guide Name]  
    **GitHub Repository:** [github.com/Thanvitha-mitta/brain-tumor](https://github.com/Thanvitha-mitta/brain-tumor)
    """)
