
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('parkinson_mri_cnn_model.h5')
model.summary()

@app.route('/predict', methods=['POST'])
def predict():
    api_key = request.headers.get('Authorization')
    if api_key != 'sua_chave_api':
        return jsonify({'error': 'Acesso não autorizado'}), 401
   
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['file']
    img = image.load_img(file, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = 'Parkinson detectado'
    else:
        result = 'Normal'
   
    return jsonify({'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True)