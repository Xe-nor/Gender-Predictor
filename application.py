from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
application = Flask(__name__)


@application.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST', 'GET'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    model_path = './face_predict_model.h5'
    # maping lables
    gender_dict = {0: 'Male', 1: 'Female'}
    # load the model
    face_model = tf.keras.models.load_model(model_path)
    # convert the image into numpy array
    img = Image.open(image_path)
    img = img.convert("L")  # converting to grayscale
    img = img.resize((128, 128))
    # # for plotting the re-sized image
    # import matplotlib as plt
    # plt.imshow(img,cmap='gray')
    np_img = np.array(img)  # converting to numpy array
    np_img = np_img[np.newaxis, :, :, np.newaxis]  # adding axes
    X = np_img.reshape(1, 128, 128, 1)
    # prediction
    pred = face_model.predict(X)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    return render_template('output.html', output1=pred_gender, output2=pred_age, imgpath=imagefile.filename)


if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0')
