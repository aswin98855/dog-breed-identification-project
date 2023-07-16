![udemy certificate](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/8cfed3bc-f1c7-4001-8414-85805f42285e)![dog-photo-3](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/4f4f52d6-5caa-499d-8c6d-84e9b1fafbf3)![dog-photo-2](https://github.com/aswin98855/dog-breed-identification-project/assets/116991167/aa2f5e39-0bde-498b-82d7-ee953a6cf4c4)# dog-breed-identification-project

This project builds an multi-class image classifier using TensorFlow 2.0 and TensorFlow Hub.

Identifying the breed of a dog given an image of a dog.

The dataset used is from Kaggle's dog breed identification.

**Some information about the data:**

We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.
There are 120 breeds of dogs (this means there are 120 different classes).
There are around 10,000+ images in the training set (these images have labels).
There are around 10,000+ images in the test set(these images have no labels, because we'll want to predict them).



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
