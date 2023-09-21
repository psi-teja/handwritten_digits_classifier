from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import matplotlib.pyplot as plt
from draw_digit import pygame, load_model_from_model_id, preprocess_image_for_inference

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr_model = load_model_from_model_id()


@app.route('/api/changemodel/<model_id>', methods=['POST'])
def change_model(model_id):
    global ocr_model
    ocr_model = load_model_from_model_id(int(model_id))
    return jsonify({'message': 'Model updated successfully'})


@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image part'})

        image_array = np.array(request.json['image'])

        grayscale_image = []

        for i in range(len(image_array)):
            if (int(i) + 1) % 4 == 0:
                grayscale_image.append(255 - image_array[i])

        grayscale_image = np.array(grayscale_image)
        dim = int(np.sqrt(len(grayscale_image)))
        grayscale_image = grayscale_image.reshape(dim, dim)

        # Save the grayscale image to the "uploads" folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
        plt.imsave(image_path, grayscale_image, cmap='gray')

        img = preprocess_image_for_inference(grayscale_image)

        digits = ocr_model.predict(img)

        pred_digit = str(digits.argmax())
        # print(digits.argmax())
        filename = f'soundtrack/{pred_digit}.wav'
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        return jsonify({'message': 'Image uploaded successfully', 'digit': pred_digit})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
