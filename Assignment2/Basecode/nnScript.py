import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

import time
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W



def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return (1 / (1 + np.exp(-z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    merged_train_data = []
    merged_train_label = []
    merged_test_data = []
    merged_test_label = []
    
    #Merging all data into train and test data respectively
    for key in mat.keys():
        yl = np.zeros((10,1), dtype= np.uintc)
        if("train" in key):
            merged_train_data.append(mat[key])
            yl[int(key[-1])] = 1
            merged_train_label.append(np.full((len(mat[key]), yl.shape[0]), yl.flatten()))
        if("test" in key):
            merged_test_data.append(mat[key])
            yl[int(key[-1])] = 1
            merged_test_label.append(np.full((len(mat[key]), yl.shape[0]),  yl.flatten()))

    #Converting list of arrays to NDarray
    merged_train_data = np.vstack(merged_train_data)
    merged_train_label = np.vstack(merged_train_label)
    test_data = np.vstack(merged_test_data)
    test_label = np.vstack(merged_test_label)

    #Feature selection, cropping zero padded area
    column_indices = []
    for i in range(784):
        if(i <= 84 or i>= 700):
            column_indices.append(i)
        if(i % 28 == 0):
            for j in range(i, i+4):
                column_indices.append(j)

    arr = np.array(range(784)).reshape(28,28)
    right_index=[]
    for i in arr:
        for j in range(4):
            column_indices.append(i[-4]+j)

    #Cropped data shape 22x20
    merged_train_data =  np.delete(merged_train_data, column_indices, axis = 1)
    test_data = np.delete(test_data, column_indices, axis = 1)
    
    all_indices = np.array(range(784))
    selected_features = np.delete(all_indices, column_indices)
    params_dict["selected_features"] = selected_features.tolist()
   
    #Splitting data randomly
    train_data_height = merged_train_data.shape[0]
    train_data_width = merged_train_data.shape[1]
    random_indices = np.random.choice(range(train_data_height), 60000, replace=False)
    train_data = np.empty((50000, train_data_width))
    train_label = np.empty((50000, merged_train_label.shape[1]), dtype=np.uintc)
    validation_data = np.empty((10000, train_data_width))
    validation_label = np.empty((10000, merged_train_label.shape[1]), dtype=np.uintc)
    for index, randi in enumerate(random_indices[:50000]):
        train_data[index] = merged_train_data[randi]
        train_label[index] = merged_train_label[randi]
    for index, randi in enumerate(random_indices[50000:]):
        validation_data[index] = merged_train_data[randi]
        validation_label[index] = merged_train_label[randi]

    print('preprocess done')
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    
    #For Scalar Error
    #Normalizing data
    training_data = np.double(training_data)/255.0
    
    #Adding bias to training data
    training_data = np.concatenate((training_data, np.ones((training_data.shape[0], 1))), axis=1)
    
    #Forward Propogation
    aj = np.matmul(w1, np.transpose(training_data))
    zj = sigmoid(aj)
    #Adding bias to zj, for the 51th node of hiddden layer
    zj = np.concatenate((zj, np.ones((1, zj.shape[1]))), axis=0)
    bj = np.matmul(w2, zj)
    ol = sigmoid(bj)
   
    #Error Function and Backward Propogation 
    n = training_data.shape[0]
    j_w1w2 = np.sum(training_label*np.log(ol).T + (1 - training_label)*(np.log(1- ol)).T, axis=None)/(-n)
    obj_val = j_w1w2 + (lambdaval/(2*n))*(np.sum(np.square(w1), axis=None) + np.sum(np.square(w2), axis=None))
    
    #For Gradient Error
    zj = zj[:-1]
    w1 = w1[:, :-1]
    w2 = w2[:, :-1]
    training_data = training_data[:, :-1]

    delta = ol.T-training_label
    dj_dw2j = (np.matmul(zj, delta).T + (lambdaval*w2))/n
    summation = np.matmul(delta, w2)
    zj_zj = np.multiply((1-zj), zj)
    zj2_summation = np.multiply(zj_zj.T, summation)
    dj_dw1jp = (np.matmul(zj2_summation.T, training_data) + (lambdaval*w1))/n
    
    dj_dw2j = np.concatenate((dj_dw2j, np.zeros((dj_dw2j.shape[0], 1))), axis=1) #zero added to last column
    dj_dw1jp = np.concatenate((dj_dw1jp, np.zeros((dj_dw1jp.shape[0], 1))), axis=1) #zero added to last column
    obj_grad = np.concatenate((dj_dw1jp.flatten(), dj_dw2j.flatten()), 0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])
    
    #Debug
    # print(f'w1 shape: {w1.shape}')
    # print(f'w2 shape: {w2.shape}')
    # print(f'training_data shape: {training_data.shape}')
    # print(f'training_label shape: {training_label.shape}')
    # print(f'zj shape: {zj.shape}')
    # print(f'bj shape: {bj.shape}')
    # print(f'ol shape: {ol.shape}')
    # print(f'j_w1w2 shape: {j_w1w2}')
    # print(f'obj_val: {obj_val}')
    # print('\n........\n')

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

    #loss function error
    aj = np.matmul(w1, np.transpose(data))
    zj = sigmoid(aj)
   
    #adding bias to zj
    zj = np.concatenate((zj, np.ones((1, zj.shape[1]))), axis=0)
    bj = np.matmul(w2, zj)
    ol = sigmoid(bj)
    labels = (ol//ol.max(0)).T
    
    return labels


"""**************Neural Network Script Starts here********************************"""
start_time = time.time()
params_dict = {}

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 80

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 40

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

print(f"\nFor lamda: {lambdaval}")
print("Training Time: %.2f seconds" % (time.time() - start_time))

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

# find the accuracy on Training Dataset
predicted_label = nnPredict(w1, w2, train_data)
print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# find the accuracy on Validation Dataset
predicted_label = nnPredict(w1, w2, validation_data)
print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# find the accuracy on Validation Dataset
predicted_label = nnPredict(w1, w2, test_data)
print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print("\nTotal Time: %.2f seconds" % (time.time() - start_time))

params_dict["n_hidden"] = n_hidden
params_dict["w1"] = w1
params_dict["w2"] = w2
params_dict["lambda"] = lambdaval
pickle.dump(params_dict, open("params.pickle", "wb"))