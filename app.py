from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from advanced_analysis import perform_analysis, predict_patient

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

last_uploaded_file = None

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
    
    global last_uploaded_file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        last_uploaded_file = filepath
        
        try:
            results = perform_analysis(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    global last_uploaded_file
    if not last_uploaded_file:
        return jsonify({'error': 'Please upload a CSV file first'}), 400
    
    data = request.json
    try:
        result = predict_patient(last_uploaded_file, data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
