import os
import cv2
from flask import Flask
from flask import render_template, request
from detector import MaskDetection


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')

@app.route("/", methods = ['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img_loc = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(img_loc)
            
            pred = predict_image(image_file)
            return render_template('index.html', prediction = pred)

    return render_template('index.html', prediction = 0)


def predict_image(img_file):
    image = cv2.imread(img_file)

    detector = MaskDetection()
    detector.preprocess_image(image)

    label, prob, bbox = detector.process_detections(image)

    return label



if __name__ == '__main__':
    app.run(debug = True)