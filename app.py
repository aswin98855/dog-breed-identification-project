from flask import Flask, render_template, request, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from PIL import Image
import uuid
import datetime
import matplotlib.pyplot as plt

# Load the trained model
with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = tf.keras.models.load_model("models/20230711-09531689069203-full-image-set-mobilenetv2-Adam.h5")
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, probability=None, image_path=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded image file
    image_file = request.files["image"]
    image_ext = os.path.splitext(image_file.filename)[1]  # Get the file extension
    unique_filename = str(uuid.uuid4()) + image_ext  # Generate a unique filename
    image_path = os.path.join("static", "images", unique_filename)  # Construct the image path

    # Save the uploaded image
    image_file.save(image_path)

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Unique breeds
    unique_breeds=['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
        'american_staffordshire_terrier', 'appenzeller',
        'australian_terrier', 'basenji', 'basset', 'beagle',
        'bedlington_terrier', 'bernese_mountain_dog',
        'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
        'bluetick', 'border_collie', 'border_terrier', 'borzoi',
        'boston_bull', 'bouvier_des_flandres', 'boxer',
        'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
        'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
        'chow', 'clumber', 'cocker_spaniel', 'collie',
        'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
        'doberman', 'english_foxhound', 'english_setter',
        'english_springer', 'entlebucher', 'eskimo_dog',
        'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
        'german_short-haired_pointer', 'giant_schnauzer',
        'golden_retriever', 'gordon_setter', 'great_dane',
        'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
        'ibizan_hound', 'irish_setter', 'irish_terrier',
        'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
        'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
        'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
        'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
        'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
        'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
        'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
        'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
        'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
        'saint_bernard', 'saluki', 'samoyed', 'schipperke',
        'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
        'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
        'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
        'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
        'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
        'west_highland_white_terrier', 'whippet',
        'wire-haired_fox_terrier', 'yorkshire_terrier']

    # Make predictions
    predictions = model.predict(image)
    predicted_index = np.argmax(predictions)
    predicted_breed = unique_breeds[predicted_index]
    probability_of_prediction = np.max(predictions) * 100
    return render_template("index.html", prediction=predicted_breed, probability=probability_of_prediction,
                        image_path=url_for('static', filename='images/' + unique_filename))

if __name__ == "__main__":
    app.run(debug=True)
