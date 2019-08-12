# CSUSB-ML-MODEL
This is a program that trains a machine learning model on a dataset of CSUSB parking lot images from a single camera. The goal is to train a model that will be able to classify the capacity of the parking lot in 10% increments.

## Getting Started

It is best practice to create a virtual environment to house the dependencies for this project. You may use an environment creator of your choice such as virtualenv, pipenv or venv. If you do not have any of these simply install one via command line. Note: You will need to have python installed on your machine.

```
pip install virtualenv
```
After you have installed virtualenv, create a python virtual environment at the preffered project location:

```
virtualenv machine_learning
```
To activate the virtual environment:

```
C:\path\to\folder\machine_learning\Scripts\activate
```
To deactivate:

```
deactivate
```

### Prerequisites

The training script will utilize many packages such as: 

```
OpenCV-Python, TensorFlow, Keras, Numpy etc.
```

### Installing
If you do not have these packages installed you can easily pip install them. They are listed in the requirements.txt file. To install all dependencies simply type into the command line of your python virtual environment:

```
pip install requirements.txt
```

Check to see if tensorflow and keras is properly installed. In the python shell type:

```
import keras as k
```

Then,
```
print(keras.__version__)
```

## Running the program

The model must first be trained using the provided dataset folder. This folder contains very few data and should be updated with more data to create a much better model.

In your command line type:

```
python train.py --dataset dataset --model parkinglot.model --labelbin parkinglot.pickle
```

The model should start training now. This will generate 2 outputs, a .model file which will hold the weights for the trained model and can be used to to make predictions. The other file is a serialized object containing the names of the "classes" that we used for training. In this case the name of the folders in the Dataset directory.

### Testing/Verifying

You can now begin testing the model. There are a few examples in the examples folder. We can test one:

```
python classify.py --model parkinglot.model --labelbin parkinglot.pickle --image examples/100.jpg
```

The test image is an image that is 100% full. The goal is to have the model identify it as such.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
