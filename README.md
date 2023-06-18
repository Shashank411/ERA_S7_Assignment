# Session 7 Assignment

## CODE 1: APPLY TECHNIQUE BATCH NORMALIZATION, REGULARIZATION, and GLOBAL AVERAGE POOLING to a Basic Skeleton

### Target:

*   Use nn.Sequential
*   Add BatchNorm
*   Apply dropOut on each layer
*   Use GAP in last layer

### Results:

*   Parameters: 5.1k
*   Best Training Accuracy: 93.31
*   Best Test Accuracy: 97.78

### Analysis:
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
  
## CODE 2: INCREASE MODEL CAPACITY AND FINE TUNE MAXPOOLING POSITION

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


## CODE 3: APPLY IMAGE AUGMENTATION AND FINE TUNE LEARNING RATE, ADD StepLR SCHEDULER

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
