from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle

# Create uploads folder if not exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and label encoder
model = tf.keras.models.load_model('pollen_model.h5')
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).resize((128, 128))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            class_name = label_encoder.inverse_transform([class_index])[0]

            return render_template('index.html', prediction=class_name, image_path=filepath)

        except Exception as e:
            return f"Error processing image: {e}"

if __name__ == '__main__':
    app.run(debug=True)