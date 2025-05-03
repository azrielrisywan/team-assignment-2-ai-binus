import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)  # Need this because Flask sets up some paths behind the scenes
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['cat','dog'] # this is what we will see in html page

@app.route("/") # by this index function will converted into flask function
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    from keras.models import load_model
    from keras.preprocessing import image
    import numpy as np

    target = os.path.join(APP_ROOT, 'images/')
    os.makedirs(target, exist_ok=True)

    upload = request.files['file']
    filename = upload.filename
    if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}):
        return "File type not allowed", 400

    destination = os.path.join(target, filename)
    upload.save(destination)

    model = load_model('cat_dog_classifier.h5')  # correct model name
    img = image.load_img(destination, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    ans = np.argmax(result, axis=1)
    confidence = round(float(np.max(result)) * 100, 2)
    prediction = "dog" if ans == 1 else "cat"

    return render_template("template.html", image_name=filename, text=prediction, confidence=confidence)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

