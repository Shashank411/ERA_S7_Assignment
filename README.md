## CODE 1: BASIC SKELETON
[CODE](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/notebooks/S5_notebook1.ipynb)

### Target:

Get the set-up right
*   Set Transforms
*   Set Data Loader
*   Set Basic Working Code
*   Set Basic Training  & Test Loop

### Results:

*   Parameters: 6.3M
*   Best Training Accuracy: 99.99
*   Best Test Accuracy: 99.24

### Analysis:

*   Extremely Heavy Model for such a problem
*   Model is over-fitting, but we are changing our model in the next step




## CODE 2: APPLY TECHNIQUE BATCH NORMALIZATION, REGULARIZATION, and GLOBAL AVERAGE POOLING
[CODE](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/notebooks/S5_notebook2.ipynb)

### Target:

*   Use nn.Sequential
*   Make model Lighter
*   Add BatchNorm
*   Apply dropOut on each layer
*   Replace last 7x7 layer with GAP

### Results:

*   Parameters: 5.1k
*   Best Training Accuracy: 93.31
*   Best Test Accuracy: 97.78

### Analysis:
*   Dropout works!
*   Model is way to much lighter
*   It seems model is under-fitting cause of very less number of parameters

  #### With BatchNorm
  - Parameters: 10 k
  - Best training accuracy = 99.88
  - Best Test accuracy = 99.26%

  #### With BatchNorm and DropOut
  - Parameters: 10 k
  - Best training accuracy = 99.04
  - Best Test accuracy = 99.13%

  #### With BatchNorm, DropOut and GAP
  - Parameters: 5.1 k
  - Best training accuracy = 93.91
  - Best Test accuracy = 97.78%
  
  
  

## CODE 3: INCREASE MODEL CAPACITY AND FINE TUNE MAXPOOLING POSITION
[CODE](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/notebooks/S5_notebook3.ipynb)

### Target:

*   Increase model capacity at the end (add layer after GAP)
*   Perform MaxPooling at RF=5 and using only one maxpooling layer

### Results:

*   Parameters: 7.9k
*   Best Training Accuracy: 99.20%
*   Best Test Accuracy: 99.39%

### Analysis:

*   Model is very good.
*   No overfitting
*   Still model is not able to get 99.4%




## CODE 4: APPLY IMAGE AUGMENTATION AND FINE TUNE LEARNING RATE, ADD StepLR SCHEDULER
[CODE](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/notebooks/S5_notebook4.ipynb)

### Target:

*   Add rotation, of (-7 to 7) degrees.
*   Add StepLR scheduler

### Results:

*   Parameters: 7.9k
*   Best training accuracy = 99.30
*   Best Test accuracy = 99.47%

### Analysis:

*   Model is awesome!!!
*   No overfittng
*   Target achieved


# FINAL MODEL ANALYSIS

## RECEPTIVE CALCULATION
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/images/RF_calculation.png)

## Loss and Accuracy plot
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/images/Loss_and_accuracy_plot.png)

## Summary
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%205%20-%20Coding%20Drill%20Down/images/logs.png)
