import os
import sys

import numpy as np

from keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image

def init():
    global model

    lib_path = os.environ.get('LIBPATH')
    if lib_path is None:
        lib_path = '../lib/'
    model_path = os.environ.get('MODELPATH')
    if model_path is None:
            model_path = '../model/'

    sys.path.append(lib_path)

    model = load_model(model_path + '/model.h5')       #3
    model.build((None, 28, 28, 1))

def infer(img_file):
    try:
        image = np.array(Image.open(img_file).convert('L')) / 255.0
        image = np.expand_dims(image, axis=0)  # Reshape to (1, 28, 28, 1)

        # 추론
        output = model.predict(image)

        # 결과 출력
        predicted = np.argmax(output)                   #2

        return None, predicted.item()     #2
    
    except Exception as e:
        return e, -1    #4

init()
app = Flask(__name__)                   #4
app.debug = True                        #4

@app.route('/recognize', methods=['POST'])          #4c
def recog_image():
    if 'image' not in request.files:                #4
        return "No image file uploaded", 400        #4
    img_file = request.files['image']             #4
    e, result = infer(img_file)

    if e == None:
        return jsonify({'result': result})
    else:
        return f"Error recognizing image: {str(e)}", 500
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)