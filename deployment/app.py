import io
import base64
import json

import numpy as np
import PIL.Image as Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image


def lambda_handler(event, context):
    image_bytes = base64.b64decode(json.loads(event["body"])["image"])  # ["body"]
    image = Image.open(io.BytesIO(image_bytes))
    image.save("/tmp/image.JPEG", "JPEG")
    print("Image saved successfully.")

    model = tf.keras.models.load_model('/saved_model.h5')

    predicted_class, predicted_probability = performInference(model, "/tmp/image.JPEG")
    print(predicted_class, predicted_probability)
    response = {
        'statusCode': 200,
        'body': f"predicted_class={predicted_class}, predicted_probability={predicted_probability}"
    }
    return response


def performInference(model, image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.xception.preprocess_input(img_array)

    # Perform inference
    predictions = model.predict(preprocessed_img)
    # Assuming the model predicts a single class
    predicted_id = np.argmax(predictions[0])
    predicted_class = "Normal" if predicted_id == 1 else "Pneumonia"
    predicted_probability = np.max(predictions[0])

    return predicted_class, predicted_probability
