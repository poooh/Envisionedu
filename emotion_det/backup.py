import os
# import keras
import face_recognition
import numpy as np
from resizeimage import resizeimage

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "/home/pooja/Desktop/Pooja.Kumari.jpg"
# rel_path = "/home/pooja/Desktop/poonew1.jpg"
#rel_path = "/home/pooja/Desktop/1_6xp-IY-M8lEEEN0UuUBq0w.jpeg"
abs_file_path = os.path.join(script_dir, rel_path)
# print(abs_file_path)
image = face_recognition.load_image_file(abs_file_path)
# print(image)
face_locations = face_recognition.face_locations(image)
# print(face_locations)
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]
# print(face_image)
# encoding_1 = face_recognition.face_encodings(image)[0]
# encoding_2 = face_recognition.face_encodings(image)[1]
# results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)
# print(results)
# import keras 
from keras.models import load_model
rel_path = "/home/pooja/Desktop/Hackgirls/face_and_emotion_detection/emotion_detector_models/model_v6_23.hdf5"
mod_file_path = os.path.join(script_dir, rel_path)
model = load_model(mod_file_path)
# print(model)
# print(face_image.shape)
# face_image = np.reshape(face_image,(,,4))
face_image = np.resize(face_image, (48,48,1))
# print(face_image.shape)
# face_image = resizeimage.resize_cover(face_image, [48, 48])
face_image = np.reshape(face_image,(1, face_image.shape[0],face_image.shape[1],face_image.shape[2]))
# print(face_image.shape)
prid_class = np.argmax(model.predict(face_image))
validation_generator = {0 : 'Angry', 5: 'Sad', 4 : 'Neutral', 1 : 'Disgust', 6 : 'Surprise', 2 : 'Fear', 3 : 'Happy'}
print(validation_generator[prid_class])