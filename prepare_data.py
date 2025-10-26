import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size (resize all images to the same size)
IMG_SIZE = (150, 150)

# Path to your dataset
train_dir = r'C:\Users\THANVITHA\Downloads\BrainTumorDataset\Training'
test_dir = r'C:\Users\THANVITHA\Downloads\BrainTumorDataset\Testing'

# Create ImageDataGenerators for training and testing datasets
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Check the class labels
print("Classes found:", train_data.class_indices)
