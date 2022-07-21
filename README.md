# ML Clothing Drawing Guesser

Draw or upload an image and the program will guess what you drew by making a machine learning model based off a sample data set

## Requirements

- Python 3+
- the following python modules
  - opencv-python https://opencv.org
  - tensorflow https://www.tensorflow.org
  - matplotlib
 
> Note: when installing tensorflow on MacOS using apple silicon make sure to install tensorflow using `pip install tensorflow-macos`
  
## Tensorflow

This program uses tensorflow to build a machine learning model that is used to predict the drawings

The dataset used is Keras fashion mnist dataset (https://keras.io/api/datasets/fashion_mnist/)
  
## OpenCV 

This program uses opencv for image manipulation as well as provides the drawing functionality for the user
  
