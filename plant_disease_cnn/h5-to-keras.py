from tensorflow.keras.models import load_model

model = load_model("D:\\ML Projects(TensorFlow)\\plant_disease_cnn\\plant_disease_cnn_model.h5")
model.save("plant_disease_cnn_model.keras")
