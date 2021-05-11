import sys
import os
import glob
import numpy as np
import pytesseract
import cv2 as cv

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT)

app = Flask(__name__)
IMAGEmodel = None

app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def model_predict(img_path, model):
    #print('----------------------------------------\n' + img_path)
    img = cv.imread(img_path)
    '''if img is None:
        print('Error durante la lectura de la imagen')
    else:
        cv.imshow('Imagen', img)
        cv.waitKey(0)'''

    img = np.asarray(img, dtype="float32")
    img = cv.resize(img, (1200, 1500))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (1500, 1200, 1))

    image_array = np.asarray([img])

    pred = model.predict(image_array)[0]

    #cv.imshow(pred)

    result = image.array_to_img(pred)
    text = pytesseract.image_to_string(result)
    print('********************************************************************************')
    print(text)
    print('********************************************************************************')

    return text 

@app.route('/predict', methods=['POST'])
def upload():
    image = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)

    # Make prediction
    text = model_predict(path, IMAGEmodel)

    return jsonify({'text': text}), 200

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

if __name__=='__main__':    	
	print("Loading IMAGE model...")
	IMAGEmodel = load_model('autoencoderModel.h5')
	print("Loaded Model from disk")
    
	app.run(host='0.0.0.0', port=5000)