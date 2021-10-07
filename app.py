from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image

import os

import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model')

def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image / 255.0)

    if result[0][0] > 0.5:
        return  'dog'
    else:
        return  'cat'







@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
          # Convert to string
        return preds
    return None

if __name__ == '__main__':
    app.run()
