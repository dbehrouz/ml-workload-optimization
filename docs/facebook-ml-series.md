# ML at Facebook

## Problem Definition

1. Determine the right task for your project
2. Simple is better than complicated
3. Define your label and training example precisely
4. Don't prematurely optimize

## Data
How to prepare and build your datasets?
Key areas that you should pay attention to:
1. Data recency and real-time training
Data is ever truly iid and time has an important role. 
The model should adapt to the underlying distribution.

2. Training/Prediction consistency
It is difficult to ensure the data used in training and prediction are consistent.
Especially, time-dependent features that vary across time.

3. Records and sampling

"Building datasets is an active part of the machine learning research."
"The choices you make in creating your training data will impact the sucess of your entire machine learning system."

## Evaluation
Model evaluation relies on two things:
- Offline evaluation using log data: it is fast and efficient for wide exploration
- Online experimentation using live traffic: is more time consuming but give a reach overview of the impact of the machine learning model

Process:
- Train
- Then evaluate on a test set
- Then analyze the predictions

First, create a baseline (simplest possible model).
Then compare the result of your more complex model with the baseline.

A good approach is to use offline evaluation to find a viable candidate and then use online experimentation to validate the candidate.
The gold standard is to split the data into three:
- Training
- validation (for hyperparameter tuning)
- Test

Another approach instead of random splitting of the data is progressive evaluation where the data is split into training, evaluation, and test is based on the time.

Older data = Training, newer data = validation, newer data = test 
This way, the results are more similar to the actual online evaluation of the machine learning model.

Useful approaches are:
- Find where the performance is coming from. Divide your data into logical groups and evaluate the model of the groups.

General rules:
1. Evaluate offline before evaluating online
2. Evaluate both the choice of data and the kind of statistics you calculate
3. Don't be bound to evaluate and train on the same thing
4. When evaluating yourmodel. understad where the performance comes from

## Features:
After data, building features is the most important aspect that affect the model.
When building features, one has to pay attention to:
- Model architecture
- Properties of the feature
- Special cases
- Training data (size)

Classes of features are categorical, continuous, and derived features.

Feature consistency between training and test is very important.
Look out for feature leakage, and test the feature coverage before spending time making the feature.

## Model
Machine learning cycle Data --> Feature --> Model -- Evaluation --> back to Data.
Practical considerations in real-world data and use cases:
- Interpretability an ease to debug
- Data volume
- Training and predicction considerations

Model settings can be divided into two groups:
1. Hyper parameters
2. Model Architecture Settings
  - Feature interactions for linear models
  - Number of leaves/trees for tree-based
  - Number, type, and width of layers for neural networks
  
This is a very large space to explore.
