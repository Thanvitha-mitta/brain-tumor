import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Automatically detect the input size from the model
input_shape = model.input_shape[1:3]  # (height, width)

print(f"Model expects input of size: {input_shape}")

# Load and preprocess the image
img_path = input("Enter the full path of the image: ")

img = tf.keras.utils.load_img(img_path, target_size=input_shape)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Assuming the image was normalized during training

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Predicted Class: {predicted_class} ({confidence * 100:.2f}% confidence)")

# Get the layer name for Grad-CAM (last convolutional layer name)
last_conv_layer_name = 'conv2d_2'  # Change this to the last convolution layer of your model

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Ensure that the model's input and output are clearly defined
    grad_model = tf.keras.models.Model(
        inputs=model.input,  # Explicitly use the model's input
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]  # Get the convolution layer output and final model output
    )
    
    with tf.GradientTape() as tape:
        # Watch the input image tensor to compute gradients
        tape.watch(img_array)
        
        # Get model predictions and the activations from the last convolutional layer
        conv_output, predictions = grad_model(img_array)
        
        predicted_class = np.argmax(predictions[0]) 
        # Compute the loss for the predicted class
        loss = predictions[:, predicted_class]
    
    # Get the gradients of the loss with respect to the convolutional output
    grads = tape.gradient(loss, conv_output)
    
    # Pool the gradients across all spatial dimensions (height, width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the convolutional output by the pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # Normalize the heatmap to make sure it's between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display the heatmap
plt.imshow(heatmap[0], cmap='jet')
plt.colorbar()
plt.show()

# Optionally, you can overlay the heatmap on the image for better visualization
# Get the original image for overlay
img = cv2.imread(img_path)
img = cv2.resize(img, (input_shape[1], input_shape[0]))  # Resize to model input size
heatmap_resized = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]))  # Resize heatmap to image size

# Apply heatmap to the image
heatmap_resized = np.uint8(255 * heatmap_resized)  # Convert heatmap to 0-255 range
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # Apply color map to heatmap
superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# Display the image with the Grad-CAM overlay
cv2.imshow("Grad-CAM", superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
