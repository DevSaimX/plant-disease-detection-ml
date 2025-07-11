from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = load_model("D:\\ML Projects(TensorFlow)\\plant_disease_cnn\\plant_disease_cnn_model.keras")

# Load class names from training folder
img_size = (128, 128)
train_dir = "D:\\ML Projects(TensorFlow)\\plant_disease_cnn\\dataset\\train"

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=1,
    class_mode='categorical'
)
class_names = list(generator.class_indices.keys())

# Load a single image for prediction
img_path = "D:\\ML Projects(TensorFlow)\\plant_disease_cnn\\dataset\\train\\Apple___Black_rot\\0b37761a-de32-47ee-a3a4-e138b97ef542___JR_FrgE.S 2908_270deg.JPG"  # Replace with your image
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]

# Extract plant and condition
plant, condition = predicted_class.split("___")
condition = condition.replace('_', ' ').capitalize()

# Display prediction
print(f"Predicted Class: {predicted_class}")
print(f"🪴 Plant: {plant}")
print(f"🩺 Diagnosis: {condition}")
