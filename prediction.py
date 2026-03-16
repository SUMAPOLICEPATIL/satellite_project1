import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = (224,224)

def predict_image(image_path, class_names):

    model = tf.keras.models.load_model("satellite_model.h5")

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    img = img/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]

    print("Predicted Land Type:", predicted_class)