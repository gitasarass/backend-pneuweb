from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, storage
from flask_cors import CORS
from config import treatment_info  # Import treatment_info from config file

app = Flask(__name__)
CORS(app)

# Update path to your service account key JSON file
cred = credentials.Certificate('path/xray-pneumonia2024-b1ef02112052.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'xray-pneumonia2024.appspot.com'
})

model = load_model('model/9191.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    return preds

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def upload():
    app.logger.info('Request received on /predict endpoint.')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = file.filename
    temp_local_path = os.path.join('static/uploads/', filename)
    file.save(temp_local_path)

    bucket = storage.bucket()
    folder_name = 'patientXray'
    blob = bucket.blob(f"{folder_name}/{filename}")
    blob.upload_from_filename(temp_local_path)
    file_url = blob.public_url

    preds = model_predict(temp_local_path, model)
    pred_percentages = preds[0] * 100

    classes = ['Normal', 'Viral', 'Bacterial', 'Covid']
    pred_results = {classes[i]: float(pred_percentages[i]) for i in range(len(classes))}

    # Retrieve treatment information based on the highest prediction
    predicted_class = classes[np.argmax(preds)]
    treatment_details = treatment_info.get(predicted_class, {})

    response = jsonify(
        {
        "prediction": pred_results,
        "img_url": file_url,
        "treatment_info": treatment_details
        }
    )

    # Allow access from all origins
    response.headers.add("Access-Control-Allow-Origin", "*")

if __name__ == '__main__':
    app.run()