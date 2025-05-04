import os
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Muat model sekali di awal
model = load_model('cat_dog_classifier_tuned.h5')
classes = ['cat', 'dog']  # Sesuai pelatihan: 0 = cat, 1 = dog

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    os.makedirs(target, exist_ok=True)

    upload = request.files['file']
    filename = upload.filename
    if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}):
        return "File type not allowed", 400

    destination = os.path.join(target, filename)
    upload.save(destination)

    # Preprocessing gambar
    img = image.load_img(destination, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0  # Normalisasi seperti pelatihan
    img = np.expand_dims(img, axis=0)

    # Prediksi
    result = model.predict(img)
    print("Raw prediction:", result)  # Debugging
    prediction = classes[np.argmax(result, axis=1)[0]]
    confidence = round(float(np.max(result)) * 100, 2)

    return render_template("template.html", image_name=filename, text=prediction, confidence=confidence)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)