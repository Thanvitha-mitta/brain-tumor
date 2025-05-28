import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('brain_tumor_model.h5')

# Class names in the same order as training
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img):
    # Process the image directly (no need to load it again)
    img = img.resize((150, 150))  # Resize the image to match the model's expected input size
    img_array = np.array(img, dtype=np.float32)  # Convert to numpy array with float32 type
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image as done during training

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return f"Predicted Class: {predicted_class} ({confidence*100:.2f}% confidence)"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,  # Function to call on image input
    inputs=gr.Image(type="pil"),  # Input component for image
    outputs="text",  # Output will be text (prediction)
)

# Launch the Gradio interface
iface.launch(share = True)
