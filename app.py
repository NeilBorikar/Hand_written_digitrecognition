from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
import base64
import re
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = keras.models.load_model("mnist_digit_recognition_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image):
    """Convert image to 28x28 grayscale for model input."""
    image = image.convert('L')                     # grayscale
    image = image.resize((28, 28))                 # resize
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 784)                  # flatten
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_digit = np.argmax(prediction[0])
    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
