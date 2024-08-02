# Deep Neural Network
This Neural Network is Trained on the MNIST Dataset of Handwritten Digits. It is made from raw python, some mathematical and utility libraries. There are no dedicated neural network libraries like tensorflow and pytorch. This was done to gain in-depth knowledge of a neural network's architecture rather than utilising pre-built models.

This code currently runs 2 epochs with a learning rate of 0.1. It has a 96% accuracy. To increase the accuracy, increasing the epoch count, or manually increasing the number of neurons in the hidden layers is needed. 

# How to Run
Simply click on main.py and run that file. If it detects a "trained_model" file, which is included by default, it will prompt the user whether they wish to train a new model and save it in the file or used the pre-trained in the file. This is done to not have to run the entire network learning process every time the program is run, but only when parameters altering the network's performance are changed.

It will output its current epoch, the average accuracy of the epoch as well as the total training time.

Once the model finishes training, or when the pre-trained model is loaded, a window is displayed with 5 test cases, displaying the image, the predicted number, and the correct number. Simply close the window to exit the program.