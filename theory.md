WEEK 1–4: Self-Driving Car in Deep Learning
1. Problem Statement – Detailed Explanation and Purpose
The objective of this experiment is to develop a self-driving car system using Deep Learning techniques. The system takes road images captured by cameras mounted on a vehicle and predicts the appropriate steering angle to control its direction. The main purpose is to train a model that can understand road conditions and make automatic steering decisions without continuous human intervention. It involves image preprocessing, steering angle analysis, model training, testing, and visualization of predicted steering angles.
2. ML and DL Models Used
- Mean Baseline Model: Predicts the average steering angle for all input images. It is used as a basic model for performance comparison.
- CNN (Convolutional Neural Network): Extracts important features such as road boundaries, curves, and lanes from images.
- NVIDIA End-to-End CNN: Uses convolutional and fully connected layers to directly predict steering angles from road images.
- Adam Optimizer: Updates model weights to minimize prediction errors.
- MSE (Mean Squared Error): Measures the difference between actual and predicted steering angles.
3. Dataset Used
Udacity Self-Driving Car Dataset is used in this experiment. It contains images captured using three cameras: center, left, and right, along with corresponding steering angles, throttle, reverse, and speed values. These images and steering values are used to train and test the CNN model.
4. Real-World Applications
- Autonomous self-driving vehicles.
- Automatic lane detection and lane keeping.
- Driver assistance systems.
- Automatic steering control.
- Intelligent transportation systems.


WEEK 5–8: Human Activity Recognition (HAR)
1. Problem Statement – Detailed Explanation and Purpose
The objective of this experiment is to develop a Human Activity Recognition (HAR) system using Machine Learning and Deep Learning techniques. The system analyzes sensor data collected from human body movements and identifies different physical activities. The main purpose is to automatically recognize activities such as walking, sitting, standing, and laying based on sensor measurements. It involves understanding the dataset, data cleaning, preprocessing, exploratory data analysis, feature extraction, model training, and performance evaluation.
2. ML and DL Models Used
Classical Machine Learning Models:
- Logistic Regression: Classifies human activities based on sensor features.
- Random Forest: Combines multiple decision trees to predict activities.
- SVM (Support Vector Machine): Separates different activity classes using an optimal decision boundary.
Deep Learning Models:
- CNN (Convolutional Neural Network): Extracts important patterns from sensor data.
- LSTM (Long Short-Term Memory): Learns sequential patterns in human movements.
3. Dataset Used
UCI Human Activity Recognition (HAR) Dataset is used in this experiment. It contains sensor data collected from 30 volunteers performing six different activities:
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying
The dataset contains 561 features extracted from accelerometer and gyroscope sensor signals. It consists of 7,352 training samples and 2,947 testing samples.
4. Real-World Applications
- Health and fitness monitoring.
- Smartwatch and smartphone activity tracking.
- Elderly people monitoring.
- Sports performance analysis.
- Smart home automation.
- Healthcare activity monitoring.


---

# gpt:

# Deep Learning Lab Viva Questions and Answers (Weeks 1–8)

The following questions are prepared by carefully referring to your uploaded **DLA LAB PDF (32 pages)**. They cover the important concepts, algorithms, models, preprocessing techniques, and implementation steps from all eight weeks.

Answers are kept simple, short, and easy to remember, with examples wherever useful.

---

# WEEK 1: Self-Driving Car Using Deep Learning

### 1. What is a Self-Driving Car?

A self-driving car is a vehicle that can move and navigate without human control by using sensors, cameras, and artificial intelligence.

**Example:** A car identifies road lanes, detects obstacles, and automatically controls its steering.

### 2. What is the objective of this experiment?

The objective is to develop a self-driving car model using a Convolutional Neural Network (CNN) to predict steering angles from road images.

**Example:** When the camera captures a road image, the model predicts whether the car should turn left, right, or move straight.

### 3. Which dataset is used in this experiment?

We use the **Udacity Self-Driving Car Dataset**.

It contains:
- Center camera images.
- Left camera images.
- Right camera images.
- Steering angles.
- Throttle values.
- Brake and speed information.

**Example:** An image of a straight road may have a steering angle close to zero.

### 4. What is steering angle prediction?

Steering angle prediction is the process of predicting the direction in which a vehicle should turn based on the input road image.

**Example:**
- Negative steering angle → Left turn.
- Positive steering angle → Right turn.
- Zero steering angle → Straight movement.

### 5. What is the difference between classification and regression?

- **Classification:** Predicts categories or classes.
- **Regression:** Predicts continuous numerical values.

**Example:**
- Classification: Predicting whether an image contains a cat or dog.
- Regression: Predicting a steering angle such as 0.25 or -0.40.

In our experiment, steering angle prediction is a regression problem.

---

# WEEK 2: Data Understanding, Dash-Cam Images and Steering Angle Analysis

### 1. What is Exploratory Data Analysis (EDA)?

EDA is the process of understanding and analyzing a dataset using statistical methods and visualizations before building a machine learning model.

**Example:** Checking the number of images, steering angle distribution, and missing values.

### 2. What are center, left, and right camera images?

These are images captured by three different cameras mounted on the front of the vehicle.

- **Center:** Captures the front road view.
- **Left:** Captures the left-side road view.
- **Right:** Captures the right-side road view.

**Example:** These images help the model understand different road positions and turning situations.

### 3. Why do we analyze steering angle distribution?

We analyze steering angles to understand how frequently the vehicle moves straight or takes turns.

It helps identify:
- Balanced and imbalanced samples.
- Frequently occurring steering angles.
- Rare turning situations.

**Example:** If most steering angles are zero, the dataset contains more straight-road images than turning images.

### 4. What is a histogram?

A histogram is a graphical representation that shows the frequency distribution of numerical data.

**Example:** A steering angle histogram shows how many images belong to each steering angle range.

### 5. Why do we balance the steering angle distribution?

We balance the dataset to reduce the dominance of frequently occurring steering angles.

If most images represent straight driving, the model may learn to predict zero steering angle most of the time.

**Example:** Removing excessive straight-driving samples helps the model learn left and right turns more effectively.

---

# WEEK 3: Deep Learning Model and NVIDIA's End-to-End CNN

### 1. What is a Deep Learning model?

Deep Learning is a branch of Machine Learning that uses artificial neural networks with multiple layers to learn patterns from data.

**Example:** A CNN learns road edges, lanes, and other visual features from driving images.

### 2. What is image preprocessing, and why is it required?

Image preprocessing is the process of modifying images before providing them to a deep learning model.

In our experiment, we perform:
- Cropping.
- RGB to YUV conversion.
- Gaussian blur.
- Image resizing.
- Pixel normalization.

**Example:** Resizing an image to 66 × 200 pixels reduces computational requirements.

### 3. What is a CNN?

CNN stands for **Convolutional Neural Network**.

It is a deep learning architecture mainly used for processing images and extracting visual features.

**Example:** CNN identifies road lanes, curves, and road boundaries from dashcam images.

### 4. What is NVIDIA's End-to-End CNN model?

NVIDIA's End-to-End CNN model is a deep learning architecture that directly maps camera images to steering angles.

It contains:
- Convolutional layers.
- Dropout layers.
- Flatten layer.
- Fully connected layers.

**Example:** Input: Road image.

Output: Predicted steering angle.

### 5. What is the purpose of a Dropout layer?

Dropout is a regularization technique that randomly disables some neurons during training to reduce overfitting.

**Example:** If the dropout rate is 0.5, approximately 50% of the selected layer's neurons are temporarily disabled during training.

---

# WEEK 4: Training, Testing and Visualization of Self-Driving Car

### 1. What is a batch in Deep Learning?

A batch is a small group of training samples processed together before updating the model weights.

**Example:** In our experiment, the batch size is 32, meaning 32 images are processed together.

### 2. What is an epoch?

An epoch is one complete pass through the entire training dataset.

**Example:** If we train a model for 30 epochs, the model processes the complete training dataset 30 times.

### 3. What is the purpose of the Adam optimizer?

Adam stands for Adaptive Moment Estimation.

It is an optimization algorithm that updates model weights using adaptive learning rates to minimize the loss.

**Example:** Adam helps the CNN learn better steering angle predictions during training.

### 4. What is Mean Squared Error (MSE)?

MSE is a loss function used to measure the average squared difference between actual and predicted values.

**Formula:**

MSE = (1/n) × Σ (Actual − Predicted)²

**Example:**

Actual steering angle = 2

Predicted steering angle = 3

MSE = (2 − 3)² = 1

A lower MSE indicates better prediction performance.

### 5. How do you evaluate the performance of the self-driving car model?

We evaluate the model by:
- Calculating test loss (MSE).
- Comparing actual and predicted steering angles.
- Visualizing steering angle predictions.
- Displaying road images with predicted steering angles.

**Example:** If the actual steering angle is 0.20 and the predicted value is 0.18, the prediction is close to the actual value.

---

# WEEK 5: Human Activity Recognition (HAR) – Problem Definition

### 1. What is Human Activity Recognition (HAR)?

Human Activity Recognition is a classification technique that identifies human activities using sensor data.

**Example:** Identifying whether a person is walking, sitting, standing, or lying down.

### 2. Which dataset is used in this experiment?

We use the **UCI Human Activity Recognition (HAR) Dataset**.

It contains sensor data collected from 30 volunteers performing six activities.

The six activities are:
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

### 3. Which sensors are used in Human Activity Recognition?

The experiment uses smartphone motion sensors:

- **Accelerometer:** Measures acceleration.
- **Gyroscope:** Measures angular velocity or rotational movement.

**Example:** When a person walks, the accelerometer detects changes in body movement.

### 4. What is the difference between time-domain and frequency-domain features?

- **Time-domain features:** Describe signal characteristics over time.
- **Frequency-domain features:** Describe the frequency components present in a signal.

**Example:**
- Time-domain: Mean and standard deviation.
- Frequency-domain: Frequency components of walking movements.

### 5. What are the applications of Human Activity Recognition?

HAR is used in:
- Health monitoring.
- Fitness tracking.
- Sports analysis.
- Smart home automation.
- Elderly people monitoring.

**Example:** A smartwatch detects whether a person is walking or running.

---

# WEEK 6: Dataset Understanding, Data Cleaning and Preprocessing

### 1. What is the purpose of dataset understanding?

Dataset understanding helps us identify the structure, size, features, labels, and data types before building a model.

In our experiment, the dataset contains:
- 7,352 training samples.
- 2,947 testing samples.
- 561 features.
- 6 activity classes.

### 2. What is data preprocessing?

Data preprocessing is the process of preparing raw data for machine learning and deep learning models.

It includes:
- Handling missing values.
- Removing duplicate records.
- Checking infinite values.
- Feature scaling.
- Label encoding.

**Example:** Converting activity labels into numerical values.

### 3. What is StandardScaler?

StandardScaler is a preprocessing technique that transforms features so they have approximately zero mean and unit standard deviation.

**Formula:**

Z = (X − Mean) / Standard Deviation

**Example:** If a feature has a large numerical range, StandardScaler brings it to a standardized scale.

### 4. What is label encoding?

Label encoding converts categorical labels into numerical values so that machine learning models can process them.

**Example:**

- WALKING → 0
- WALKING_UPSTAIRS → 1
- WALKING_DOWNSTAIRS → 2
- SITTING → 3
- STANDING → 4
- LAYING → 5

### 5. What is data leakage, and how can we avoid it?

Data leakage occurs when information from the testing dataset is unintentionally used during model training.

We avoid it by fitting preprocessing techniques only on the training data.

**Example:** Fit StandardScaler using training data and use the same scaler to transform testing data.

---

# WEEK 7: Exploratory Data Analysis – Univariate Analysis

### 1. What is Univariate Analysis?

Univariate analysis is the process of analyzing one variable at a time to understand its distribution and characteristics.

**Example:** Analyzing the frequency of walking, sitting, and standing activities.

### 2. What is activity distribution?

Activity distribution shows the number of samples belonging to each activity class.

**Example:** If walking has 1,400 samples and laying has 1,000 samples, walking has more samples than laying.

### 3. Why do we use a bar chart?

A bar chart is used to compare the frequency or count of different categories.

**Example:** Comparing the number of samples in six human activities.

### 4. What is a pie chart?

A pie chart represents the percentage contribution of different categories to the total dataset.

**Example:** Showing what percentage of the dataset belongs to walking, sitting, and standing.

### 5. Why do we use a histogram in EDA?

A histogram helps us understand the distribution and frequency of numerical features.

**Example:** Plotting the distribution of the tBodyAcc-mean()-X feature to understand its values.

---

# WEEK 8: Classical Machine Learning and Deep Learning Models

### 1. What is the difference between Classical Machine Learning and Deep Learning?

**Classical Machine Learning:**
- Uses algorithms such as Logistic Regression, Random Forest, and SVM.
- Often requires feature extraction or preprocessing.

**Deep Learning:**
- Uses neural networks such as CNN and LSTM.
- Can learn complex patterns automatically.

**Example:** Random Forest is a classical ML algorithm, whereas CNN is a deep learning model.

### 2. What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for classification problems.

It predicts the probability of an input belonging to a particular class.

**Example:** Predicting whether a person is walking or sitting based on sensor features.

### 3. What is Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make a final prediction.

**Example:** Multiple decision trees predict a person's activity, and the majority prediction becomes the final output.

### 4. What is Support Vector Machine (SVM)?

SVM is a supervised machine learning algorithm that finds the best decision boundary (hyperplane) to separate different classes.

**Example:** Classifying sensor data into walking and sitting activities.

### 5. What is the difference between CNN and LSTM?

- **CNN (Convolutional Neural Network):** Extracts important patterns and features from input data.
- **LSTM (Long Short-Term Memory):** Learns sequential patterns and remembers information over time.

**Example:**
- CNN: Identifying patterns in sensor features.
- LSTM: Understanding the sequence of movements while walking.

---

## Quick Revision – Important Concepts to Remember

| Week | Important Concepts |
|---|---|
| Week 1 | Self-driving car, Udacity dataset, Regression |
| Week 2 | EDA, Dashcam images, Steering angles, Histogram |
| Week 3 | Image preprocessing, CNN, NVIDIA architecture, Dropout |
| Week 4 | Batch, Epoch, Adam optimizer, MSE |
| Week 5 | HAR, UCI dataset, Accelerometer, Gyroscope |
| Week 6 | Data cleaning, StandardScaler, Label Encoding, Data Leakage |
| Week 7 | Univariate analysis, Bar chart, Pie chart, Histogram |
| Week 8 | Logistic Regression, Random Forest, SVM, CNN, LSTM |

**PDF reference:** All eight weeks are covered in your uploaded 32-page lab manual. The self-driving car experiments are documented on pages 1–18, and Human Activity Recognition experiments are documented on pages 19–32. DLA LAB_1(1)

**Viva preparation tip:** Give priority to understanding the dataset, objective, algorithm, input, output, and purpose of each technique. These are the fundamental concepts behind all eight experiments.

---

If you want, I can:

- Generate detailed explanation for NVIDIA’s end-to-end CNN model
- Create example questions on data preprocessing in Human Activity Recognition
- Suggest practical steps to validate self-driving car model performance