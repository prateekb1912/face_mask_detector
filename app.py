import os
from flask import Flask
from flask import render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')

@app.route("/", methods = ['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img_loc = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(img_loc)
            return render_template('index.html', prediction = 1)        
    return render_template('index.html', prediction = 0)

if __name__ == '__main__':
    app.run(debug = True)