import os
# import keras
import face_recognition
from face_recognition import api
import numpy as np
from werkzeug.utils import secure_filename
from skimage.color import rgb2gray
# from resizeimage import resizeimage

from flask import Flask, request, render_template

UPLOAD_FOLDER = '/home/pooja/Desktop/Hackgirls/emotion_det/static'
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import skimage.io as io
import skimage.transform as trans

def read_image(img_path, target_size = (48, 48), as_gray = True):
	img = io.imread(img_path, as_gray = as_gray)
	img = img / 255
	img = trans.resize(img,target_size)
	img = np.reshape(img,img.shape+(1,))
	img = np.reshape(img,(1,)+img.shape)

	return img

def sanitize_image(img, target_size = (48, 48)):
	print("into satistize")
	img = rgb2gray(img)
	print(img.shape)
	img = trans.resize(img,target_size)
	print(img.shape)
	img = np.reshape(img,img.shape+(1,))
	print(img.shape)
	img = np.reshape(img,(1,)+img.shape)
	print(img.shape)

	return img


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/emotion", methods=["GET"])
def emotion():
	return render_template('emotion.html')

@app.route("/get_emotion", methods=["GET", "POST"])
def get_emotion():
	print("********inside**********")
	file = request.files['file']
	print(file)
	print("********check1**********")
	filename = ""
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		print(filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	else:
		filename = "Pooja.Kumari.jpg"
	# # sfname = 'images/'+str(secure_filename(f.filename))
	# sfname = 'static/images/'+str(secure_filename(f.filename))
	# f.save(sfname)
	# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
	# rel_path = "/home/pooja/Desktop/Pooja.Kumari.jpg"
	# rel_path = "/home/pooja/Desktop/poonew1.jpg"
	#rel_path = "/home/pooja/Desktop/1_6xp-IY-M8lEEEN0UuUBq0w.jpeg"
	# abs_file_path = os.path.join(script_dir, rel_path)
	abs_file_path = os.path.join(UPLOAD_FOLDER, filename)
	print("********check2**********")
	print(abs_file_path)
	image = api.load_image_file(abs_file_path)

	# image = read_image(abs_file_path)
	# print(image)
	print("********check4**********")
	print(image.shape)
	face_locations = face_recognition.face_locations(image)
	# print(face_locations)
	top, right, bottom, left = face_locations[0]
	face_image = image[top:bottom, left:right]
	print("********check3**********")

	print(face_image.shape)
	print("########check 5#######")
	print(sanitize_image(face_image).shape)
	face_image = sanitize_image(face_image)
	# encoding_1 = face_recognition.face_encodings(image)[0]
	# encoding_2 = face_recognition.face_encodings(image)[1]
	# results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)
	# print(results)
	# import keras 
	from keras.models import load_model
	rel_path = "/home/pooja/Desktop/Hackgirls/face_and_emotion_detection/emotion_detector_models/model_v6_23.hdf5"
	# mod_file_path = os.path.join(rel_path)
	# print(mod_file_path)
	model = load_model(rel_path)
	# print(model)
	# print(face_image.shape)
	# face_image = np.reshape(face_image,(,,4))
	# face_image = np.resize(face_image, (48,48,1))
	# print(face_image.shape)
	# face_image = resizeimage.resize_cover(face_image, [48, 48])
	# face_image = np.reshape(face_image,(1, face_image.shape[0],face_image.shape[1],face_image.shape[2]))
	# print(face_image.shape)
	prid_class = np.argmax(model.predict(face_image))
	validation_generator = {0 : 'Angry', 5: 'Sad', 4 : 'Neutral', 1 : 'Disgust', 6 : 'Surprise', 2 : 'Fear', 3 : 'Happy'}
	print(validation_generator[prid_class])
	return validation_generator[prid_class]


if __name__ == '__main__':
    app.run(host='128.235.159.0', port=80)

# @app.route("/get_rec", methods=["GET"])
# def get_rec():
# 	pass
