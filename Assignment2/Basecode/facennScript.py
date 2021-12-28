'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize

import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return (1 / (1 + np.exp(-z)))
    
    
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

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1))) #with bias
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1))) #with bias
    obj_val = 0

    # Your code here

    # For Scalar Erorr
    # Converting ndarray to matrix for handling 1d-array transpose 
    training_label = np.matrix(training_label).T
    
    #Adding bias to training data
    training_data = np.concatenate((training_data, np.ones((training_data.shape[0], 1))), axis=1)

    #Forward Propogation
    aj = np.matmul(w1, np.transpose(training_data))
    zj = sigmoid(aj)
    #adding bias to zj, for the 257th node of hiddden layer
    zj = np.concatenate((zj, np.ones((1, zj.shape[1]))), axis=0)
    bj = np.matmul(w2, zj)
    ol = np.matrix(sigmoid(bj))

    #Error Function and Backward Propogation 
    n = training_data.shape[0]
    j_w1w2 = np.sum((np.multiply(training_label, np.log(ol).T)) + (np.multiply((1 - training_label), (np.log(1- ol)).T)), axis=None)/(-n)
    obj_val = j_w1w2 + (lambdaval/(2*n))*(np.sum(np.square(w1), axis=None) + np.sum(np.square(w2), axis=None))

    #For Gradient Error
    zj = zj[:-1]
    w1 = w1[:, :-1]
    w2 = w2[:, :-1]
    training_data = training_data[:, :-1]

    delta = ol.T-training_label
    dj_dw2j = (np.matmul(zj, delta).T + (lambdaval*w2))/n
    summation = np.matmul(delta, w2) #21100x256
    zj_zj = np.multiply((1-zj), zj) #256x21100
    zj2_summation = np.multiply(zj_zj.T, summation) #21100x256
    dj_dw1jp = (np.matmul(zj2_summation.T, training_data) + (lambdaval*w1))/n

    dj_dw2j = np.array(dj_dw2j)
    dj_dw1jp = np.array(dj_dw1jp)

    dj_dw2j = np.concatenate((dj_dw2j, np.zeros((dj_dw2j.shape[0], 1))), axis=1) #zero added to last column
    dj_dw1jp = np.concatenate((dj_dw1jp, np.zeros((dj_dw1jp.shape[0], 1))), axis=1) #zero added to last column
    obj_grad = np.concatenate((dj_dw1jp.flatten(), dj_dw2j.flatten()), 0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    #debug
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

    # Your code here
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

    #Loss function error
    aj = np.matmul(w1, np.transpose(data))
    zj = sigmoid(aj)

    #Adding bias to zj
    zj = np.concatenate((zj, np.ones((1, zj.shape[1]))), axis=0)
    bj = np.matmul(w2, zj)
    ol = sigmoid(bj)

    #For n_class = 1
    threshold = np.empty(ol.shape)
    threshold.fill(0.5263)
    labels = (ol//threshold).astype(int)
   
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
start_time = time.time()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 1

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 55;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

print(f"\nFor lamda: {lambdaval}")
print("\nTraining Time: %.2f seconds" % (time.time() - start_time))

#Test the computed parameters

#find the accuracy on Training Dataset
predicted_label = nnPredict(w1,w2,train_data)
print('\nTraining set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#find the accuracy on Validation Dataset
predicted_label = nnPredict(w1,w2,validation_data)
print('Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#find the accuracy on Validation Dataset
predicted_label = nnPredict(w1,w2,test_data)
print('Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print("\nTotal Time: %.2f seconds" % (time.time() - start_time))
