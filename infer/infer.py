import os

import numpy as np

from keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image

lib_path = os.environ.get('LIBPATH')
model_path = os.environ.get('MODELPATH')

import sys
sys.path.append(lib_path)

app = Flask(__name__)                   #4
app.debug = True                        #4

@app.route('/recognize', methods=['POST'])          #4
def recog_image():
    if 'image' not in request.files:                #4
        return "No image file uploaded", 400        #4
    f = request.files['image']             #4
    try:
        model = load_model(model_path + '/model.h5')       #3
        model.build((None, 28, 28, 1))

        image = np.array(Image.open(f).convert('L')) / 255.0
        image = np.expand_dims(image, axis=0)  # Reshape to (1, 28, 28, 1)

        # 추론
        output = model.predict(image)

        # 결과 출력
        predicted = np.argmax(output)                   #2

        return jsonify({'result':predicted.item()})     #2
    
    except Exception as e:
        return f"Error recognizing image: {str(e)}", 500    #4

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)