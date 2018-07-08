# My_Traffic_Sign_Classifier

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

### Submission Files

This project includes

- The notebook `Traffic_Sign_Classifier_My_Project.ipynb` (and `signames.csv` for completeness)
- `report.html`, the exported HTML version of the python notebook
- A directory `test_images` containing images found on the web
- `README.md`, which you're reading

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27911.
* The size of the validation set is 6978.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because signs in our training set are differentiable from their
contents and shapes, and the network seems having no problem to learn just from shapes.

Then I normalized the images as usual.

But after having a look at the dataset, I realized that the dataset was too much specific on the situation under which the pictures were taken. As a result, I decided to add 4 randomizing function for the preprocessing: random_translate, random_scaling, random_warp and random_brightness.


The difference between the original data set and the augmented data set is shown in cell 18. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400,  outputs 120        									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| 120,  outputs 84        									|
| RELU					|												|
| Dropout				|												|
| Softmax				|        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using following hyperparameters: learning rate of 9e-4 , dropout rate of 0.5 and batch size of 100.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%.
* validation set accuracy of 98.2%. 
* test set accuracy of 92.1%.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Initial training experiments a grid search approach with learning rate 0.001, and a batch size of 128. These parameters where chosen since they are typical values for training convnets for image recognition. The larger batch size (rather than 64 for LeNet lab) can lead to less noisy updates of the weights. In most of the simulations I run the experiments for 25 epochs, however, in the final rounds I changed it to 60 epochs to see if any improvement could be made. It appears that the model objective saturates fairly quickly and hence, the increased number of epochs do not help. This is to be expected with an increased batch size, however.

 

### Test a Model on New Images

#### 1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign160.png) ![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign170.png) ![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign180.png)
![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign190.png) ![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1910.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1920.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1930.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1940.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1950.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign1960.png)![ ](https://github.com/wang422003/My_Traffic_Sign_Classifier/blob/master/test_images/sign90.png)

The first image might be difficult to classify because the sensitometry was too low and the sign was difficult to be recognized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (80km/h)      		| Speed limit (80km/h)   									| 
| Road work     			| Road work 										|
| Speed limit (50km/h)      		| Speed limit (50km/h)   									| 
| Keep right					| Keep right										|
| Priority road					| Priority road									|
| Ahead only				| Ahead only									|
| Speed limit (60km/h)					| Speed limit (60km/h)									|
| Keep right					| Keep right										|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection					 				|
| General caution			| General caution     							|
| Priority road				| Priority road	     							|


The model was able to correctly guess 11 of the 11 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.1%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 37th cell of the Ipython notebook.

The softmax probabilities for all the images are 100% on the true one, which proofs that the model is pretty good.


