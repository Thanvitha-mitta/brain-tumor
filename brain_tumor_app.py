# brain_tumor_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

TARGET_SIZE = (150, 150)

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_brain_tumor_model(path="brain_tumor_model.h5"):
    model = load_model(path, compile=False)
    dummy = np.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    try:
        _ = model(dummy)
    except Exception:
        pass
    return model

model = load_brain_tumor_model()

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
FRIENDLY_LABELS = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

# ------------------ GRAD-CAM FUNCTION ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    import tensorflow as tf
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No convolutional layer found.")
    if last_conv_layer_name not in [l.name for l in model.layers]:
        last_conv_layer_name = conv_layers[-1]

    new_input = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    x = new_input
    cloned_outputs = []
    for old_layer in model.layers:
        try:
            config = old_layer.get_config()
            LayerClass = old_layer.__class__
            new_layer = LayerClass.from_config(config)
            try:
                new_layer.build(x.shape)
                new_layer.set_weights(old_layer.get_weights())
            except Exception:
                pass
        except Exception:
            new_layer = old_layer
        try:
            if 'training' in new_layer.call.__code__.co_varnames:
                x = new_layer(x, training=False)
            else:
                x = new_layer(x)
        except Exception:
            x = new_layer(x)
        cloned_outputs.append(x)

    idx = [l.name for l in model.layers].index(last_conv_layer_name)
    cloned_conv_output = cloned_outputs[idx]
    grad_model = Model(inputs=new_input, outputs=[cloned_conv_output, x])

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        top_channel = predictions[:, pred_index]
    grads = tape.gradient(top_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.45):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

def preprocess_pil(img):
    arr = img.resize(TARGET_SIZE)
    arr = np.array(arr).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0)

# ------------------ PREDICT ------------------
def predict(img_pil):
    x = preprocess_pil(img_pil)
    preds = model.predict(x)
    preds = preds[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx], preds, x

# ------------------ PDF REPORT ------------------
def generate_pdf(pred_label, conf, overlay_rgb):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, height - 50, "🧠 Brain Tumor Diagnosis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Prediction: {FRIENDLY_LABELS.get(pred_label, pred_label)}")
    c.drawString(50, height - 120, f"Confidence: {conf*100:.2f}%")

    # Convert overlay RGB -> JPEG for report
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    Image.fromarray(overlay_rgb).save(tmp.name, "JPEG")
    c.drawImage(ImageReader(tmp.name), 50, height - 500, width=500, height=350)
    tmp.close()

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Generated automatically by Explainable AI System")
    c.save()
    buffer.seek(0)
    os.remove(tmp.name)
    return buffer

# ------------------ UI ------------------
st.title("🧠 Brain Tumor Classifier — Explainable AI")
st.write("Upload a brain MRI. The system predicts the tumor type, confidence, and highlights regions used in prediction.")

uploaded = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Analyze MRI"):
        with st.spinner("Analyzing... Please wait ⏳"):
            pred_label, conf, preds, x = predict(img_pil)
            friendly = FRIENDLY_LABELS[pred_label]
            st.markdown(f"## 🩺 Prediction: **{friendly}**")
            st.markdown(f"**Confidence:** {conf*100:.2f}%")

            if pred_label == "notumor":
                st.success("No tumor detected — great news! 🎉🎈")
                st.balloons()
            else:
                st.warning("Tumor detected — please consult a specialist. ⚠️")

            # Interactive Plotly bar
            st.subheader("Prediction Confidence by Class")
            fig = go.Figure(
                go.Bar(
                    x=[FRIENDLY_LABELS[c] for c in CLASS_NAMES],
                    y=preds * 100,
                    marker_color=[
                        "#1E88E5" if c == pred_label else "#B0BEC5" for c in CLASS_NAMES
                    ],
                    text=[f"{p*100:.2f}%" for p in preds],
                    textposition="outside",
                )
            )
            fig.update_layout(
                yaxis_title="Confidence (%)",
                xaxis_title="Tumor Type",
                template="plotly_white",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Grad-CAM visualization
            st.subheader("Grad-CAM Explainability")
            try:
                convs = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
                last_conv = convs[-1] if convs else None
                heatmap = make_gradcam_heatmap(x, model, last_conv)
                base = np.array(img_pil.resize(TARGET_SIZE))
                base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
                overlay_bgr = overlay_heatmap(heatmap, base_bgr)
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, caption="Grad-CAM overlay (model attention)", use_container_width=True)

                # Generate downloadable PDF
                pdf_buf = generate_pdf(pred_label, conf, overlay_rgb)
                st.download_button(
                    label="📄 Download Diagnosis Report (PDF)",
                    data=pdf_buf,
                    file_name="brain_tumor_report.pdf",
                    mime="application/pdf",
                )

            except Exception as e:
                st.error(f"Grad-CAM generation failed: {e}")

else:
    st.info("Please upload an MRI image to start diagnosis.")

st.markdown("---")
st.caption("Explainable AI for Brain Tumor Diagnosis")
