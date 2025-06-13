from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gdown
# Initialize Flask app
app = Flask(__name__)

# Model download if not exists
MODEL_PATH = 'model/plant_disease_prediction_model.h5'
if not os.path.exists(MODEL_PATH):
    os.makedirs('model', exist_ok=True)
    url = 'https://drive.google.com/file/d/1-31V5KyRvlLpLcBgRJWfj18FMLWm_vIV/view?usp=sharing'  # replace with real ID
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# Classes from PlantVillage (example - update based on your model’s actual training labels)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'corn___northern_Leaf_Blight',
    'Corn___Common_rust',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',

    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',

    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Target_Spot',
    'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___mosaic_virus',
    'Tomato___healthy'
]

# Preprocessing function
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# API route to make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save and preprocess the image
    file_path = os.path.join('temp.jpg')
    file.save(file_path)
    input_data = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(prediction[0]))

    # Delete the temporary file
    os.remove(file_path)

    return jsonify({
        'prediction': predicted_class,
        'confidence': f"{confidence * 100:.2f}%"
    })

# Health check route
@app.route('/')
def home():
    return 'Plant Disease Prediction API is running!'

if __name__ == '__main__':
    app.run(debug=True)

