
# 🌿 Plant Disease Detection Project

This project detects plant diseases using a Convolutional Neural Network (CNN) and a Streamlit app.

---

## 📁 Folder Structure
```
plant_disease_cnn/
├── dataset/
│   ├── train/
│   └── test/
├── app.py
├── plant_disease_cnn_model.h5
├── requirements.txt
├── prediction_log.csv (optional)
├── .gitignore
└── README.md
```

---

## 🧠 Model Training

### 📦 Data Preparation
The dataset is organized into `train/` and `test/` folders with subfolders for each class.

### 🏗️ CNN Architecture (in `train_model.py`)
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
```

Trained using:
```python
model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,
    callbacks=[EarlyStopping(patience=3)]
)
```

Saved model:
```python
model.save("plant_disease_cnn_model.h5")
```

---

## 🧪 Prediction Script
`predict.py` loads a test image and prints:
```python
# Load model and predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
plant, condition = predicted_class.split("___")
print(f"🪴 Plant: {plant}, 🩺 Condition: {condition}")
```

---

## 🌐 Streamlit App (`app.py`)
Key features:
- Upload any image for diagnosis
- Display prediction and class name
- Logs history in `prediction_log.csv`
- Shows recent predictions at the bottom

---

## ☁️ Deployment on Streamlit Cloud
1. Upload files to GitHub repo
2. Ensure `requirements.txt` includes:
```
streamlit
tensorflow
pandas
numpy
matplotlib
```
3. Deploy from https://streamlit.io/cloud with `app.py` as the entry point

---

## ✅ Optional Enhancements
- Add image preview with prediction log
- Use SQLite/Google Sheets for persistent logs

---

🎉 Done! Your app is ready for global use.
