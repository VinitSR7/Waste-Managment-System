import os
from flask import Flask, flash, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from predictor_api import make_prediction

UPLOAD_FOLDER = './static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#print(make_prediction("uploads/garbage.jpeg"))
#print(predict("garbage.jpeg"))
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print(make_prediction("static/uploads/garbage.jpeg"))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('index.html')

from flask import send_from_directory

@app.route('/camera', methods=['GET'])
def camera():
  return render_template('camera.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    image =  filename
    msg = make_prediction(image)
    return render_template('result.html', msg=msg, image=image )

@app.route('/predict', methods=['GET'])
def predict():
    m = make_prediction("garbage.jpeg")
    print(m)
    if(m == 1):
        return '''
        true '''

    return ''' false '''


if __name__=="__main__":
    app.run(debug=True)
    app.run()
