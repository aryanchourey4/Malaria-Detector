from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import keras

app = Flask(__name__)

model = keras.models.load_model('my_model.h5')

dic = {0: 'Parasitized', 1: 'Uninfected'}


def model_predict(img_path, model):
  i = load_img(img_path, target_size=(64, 64))

  i = np.array(i)
  i = i.reshape(1, 64, 64, 3)
  p = model.predict(i)
  print(p)
  return p


@app.route('/', methods=['GET'])
def index():
  # Main page
  return render_template('index.html')


import os


@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'uploads')
    if not os.path.exists(upload_path):
      os.makedirs(upload_path)
    file_path = os.path.join(upload_path, secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    pred_class = model_predict(file_path, model)
    pred_class = np.argmax(pred_class[0])
    # Convert class index to class label

    result = str(dic[pred_class])
    return result
  return None


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
