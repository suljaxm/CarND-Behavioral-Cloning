# Behavioral-Cloning
The network is to predict control the steering angle of the car through the camera image. 

--- 
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Net.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_36_31_472.jpg "Grayscaling"
[image3]: ./examples/left_2016_12_01_13_36_31_472.jpg "Recovery Image_left"
[image4]: ./examples/right_2016_12_01_13_36_31_472.jpg "Recovery Image_right"
[image5]: ./examples/center_image_flipped.jpg "Flipped Image"
[image6]: ./examples/center_image_cropped.jpg "Cropped Image"
[image7]: ./examples/result.png "Result Image"
[image8]: ./examples/result_track2.png "Result_track2 Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**ps**: model.h5 download from [here](https://pan.baidu.com/s/1YeFK52BrgxiT6e7xoC6ZQQ) and the password is *1zow*
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. I follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. Those were all inspired by NVIDIA’s [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper (model.py lines 77-110). This paper proposed a network to map raw pixels from a single front-facing camera directly to steering commands. So this paper is consistent with the purpose of this project.

NVIDIA's work is the design based on convolutions. According to the author, they designed the convolutional layers act as a feature extractor, whereas the fully connected layers are in charge of the control for the steering angle. However, as they note, the system end-to-end is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.


#### 2. Attempts to reduce overfitting in the model

The model contains pooling layers in order to reduce overfitting (model.py lines 84,87,92,95,98). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 12). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road steering angle due to the *correction*  (code line 43).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to control the steering angle of the car through the camera image. 

My first step was to use a convolution neural network model similar to the *LeNet*. I thought this model might be appropriate because it is a mature classification network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that the net could reduce overfitting. Then I added dropout layer, but the effect is not ideal. After that, I borrowed from NVIDIA's work.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I shuffled the training order of the dataset and increased the number of batches. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-110) consisted of a convolution neural network with the following layers and layer sizes,

| Layer         		|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 90x320x3 RGB image   		| 
| Convolution 5x5(RELU)     	| 2x2 stride, same padding, outputs 45x160x24	|
| Max pooling	      	| 1x1 stride,  outputs 44x159x24	|
| Convolution 5x5(RELU)	   | 2x2 stride, same padding, outputs 22x80x36    |
| Max pooling	      	| 1x1 stride,  outputs 21x79x36	|
| Convolution 5x5(RELU) | 2x2 stride, same padding, outputs 11x40x48    |
| Max pooling	      	| 1x1 stride,  outputs 10x39x48	|
| Convolution 3x3(RELU)  | 1x1 stride, same padding, outputs 10x39x64    |
| Max pooling	      	| 1x1 stride,  outputs 9x38x48	|
| Convolution 3x3(RELU) | 1x1 stride, same padding, outputs 9x38x64    |
| Max pooling	      	| 1x1 stride,  outputs 8x37x48	|
| Fully connected(RELU)| inputs 18944, outputs 1164    |
| Fully connected(RELU)| inputs 1164, outputs 100    |
| Fully connected(RELU)| inputs 100, outputs 50    |
| Fully connected(RELU)| inputs 50, outputs 10    |
| Fully connected		| inputs 10, outputs 1    |

At first, I adopted the dropout strategy, and found that it was easy to underfit and the model was not easy to converge, so I chose the pooling layer with the stride of 1x1. Meanwhile, the convolution with a stride of 1x1 is used to retain as much predictive information as possible.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I downloaded the relevant [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) from the Internet. Here is an example image of center lane driving:

![alt text][image2]

Because of the simulator will simultaneously save an image for the left, center and right cameras, I then changed the steering measurement based on the center camera as a new measurement for the left and right cameras (model.py line 43,46,51).

 ![alt text][image4] ![alt text][image2] ![alt text][image3] 

The steering measurement from left to right are *0.3383082*, *0.1383082* and  *-0.1383082*.

To augment the data sat, I also flipped images and angles thinking that this would improve the stability of network prediction. For example, here is an image that has then been flipped (model.py line 38,39).

![alt text][image2] ![alt text][image5]

By the way, not all of these pixels contain useful information, however. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. 
I used*Cropping2D* to crop the image.  For example, here is an image that has then been cropped (model.py line 79).

![alt text][image2] ![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by visualizing loss. 

![alt text][image7]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Conclusions
The parameters that need to be adjusted are 
* **batch_size=32**, 
* **correction = 0.2**, 
* **cropping=((50,20), (0,0))**
* **train_test_split(samples, test_size=0.2)**
*  Dataset shuffle processing is also important.
*  The more training data, the better the effect of training, which I experienced in Treak2. I found that the training effect was completely different between the data collected in one circle and the data collected in multiple circles.


