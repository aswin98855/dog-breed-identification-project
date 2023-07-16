# dog-breed-identification-project

## (The clear explanation of Training Model is clearly explained in Google Colab Notebooks attached in this repository. Another colab notebook is attached which is shortform of the training model)

This project builds an multi-class image classifier using TensorFlow 2.0 and TensorFlow Hub.

Identifying the breed of a dog given an image of a dog.

The dataset used is from Kaggle's dog breed identification.

**Some information about the data:**

We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.

There are 120 breeds of dogs (this means there are 120 different classes).

There are around 10,000+ images in the training set (these images have labels).

There are around 10,000+ images in the test set(these images have no labels, because we'll want to predict them).

## Creating Model (Mobilenet v2)

![p5](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/800f8c77-89f3-424c-bc9c-1118256ba809)

## Training Model

![p4](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/41502934-c524-4af3-9731-745b6c45fcdd)

## Opening Page of Dog Breed Prediction

![p1](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/3f170873-3b3e-4e0a-b355-226a1ed9e646)

## Dog Breed Prediction

![p3](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/70717a90-f2e1-4bcc-87a9-5be3a6df1924)

## Prediction Probability Representation

![p2](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/bbdd9eaa-eb83-4766-9cbf-afe3083b770c)

# Set up for flask :

If you are using Visual Studio Code to run your Flask application locally, you can follow these steps:

Open your project folder in Visual Studio Code.


Make sure you have the required dependencies installed. You can use a virtual environment to keep the dependencies isolated. 

To create a virtual environment, open a terminal in Visual Studio Code and run the following commands:


`python -m venv venv`  # Create a virtual environment

`source venv/bin/activate`  # Activate the virtual environment (for macOS/Linux)

`.\venv\Scripts\activate`  # Activate the virtual environment (for Windows)

**Install the necessary packages by running the following command in the terminal:**

`pip install flask tensorflow`


**To install the tensorflow_hub and PIL (Python Imaging Library) packages in your virtual environment, you can use the following commands:**

**For tensorflow_hub:**

`pip install tensorflow_hub`

**For PIL:**

`pip install pillow`

Make sure you have activated your virtual environment before running these commands. Once installed, you can import and use the packages in your Flask application.

Run the Flask application by executing the following command in the terminal:

`python app.py`

The Flask application will start, and you can access it in a web browser by navigating to `http://localhost:5000`

Open a web browser and navigate to the URL displayed in the terminal to access the Flask application.

You should now be able to upload an image and receive the predicted dog breed.
