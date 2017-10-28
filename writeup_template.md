#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Model Visualization"
[image6]: ./examples/original.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 and fully connected layers with dropout layers between them.(model.py lines 57-71) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 66,68 and 70). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

####4. Appropriate training data
I used the training data provided by udacity to train the model. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an established model. Add layers if needed for overfitting. Tune the model parameters to get the best model.

My first step was to use a convolution neural network model similar to the one used by Nvidia team. I thought this model might be appropriate because the nvidia team has tested it and also suggested by Paul Hearty  (https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) mentioned in useful resources for this project,

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include 3 dropout layers so that there the training data and validation data has similar mean squared errors.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell on the yellow line to improve the driving behavior in these cases, I tested around with different values of dropout rate, angle correction and epochs to get the best model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 57-71) consisted of a convolution neural network with the following layers and layer sizes 
The final model architecture is a convlutional neural network with the following layers:

1) Convolution 2D layer with 5x5 size
2) Convolution 2D layer with 5x5 size
3) Convolution 2D layer with 5x5 size
4) Convolution 2D layer with 3x3 size
4) Convolution 2D layer with 3x3 size
5) Flatten layer
6) Dense with 100 elements
7) Dropout with 50% chance
8) Dense with 50 elements
9) Dropout with 50% chance
10) Dense with 10 elements
11) Dropout with 30% chance
12) Dense with 1 element for regression.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process
I used the data provided by udacity. I recorded multiple laps for track 1 and 2 in both forward and backward direction but every time I used that data and even changed the parameters for the model, the model became worse. The main reason was not having smooth driving on the tracks as it was very difficult to comtrol the car with the keyboard. Also the turn angles were extreme leading to worsening of model. I plan get a joystick and give it a go again later.

To augment the data sat, I also flipped images thinking that this would be give data for a track mirrored to the one being driven on and will have no negative effect on the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 48,216 number of data points. I then preprocessed this data by normalizing the data and also cropping the images to remove the cars front and the sky including trees etc and preserving the track in the images. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by increase in validation loss after 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The video for the track 1 can be seen in Track1.mp4
