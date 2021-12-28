import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    class_array = np.unique(y)
    features_len = X.shape[1]
    class_len = len(class_array)

    indices = []
    for i in range(class_len):
        indices.append(np.where(y==class_array[i])[0])

    class_sum = []
    for i in indices:
        total = 0;
        for j in i:
            total += X[j]
        class_sum.append(total)

    lengths = []
    for i in indices:
        lengths.append(len(i))


    means = np.ones((class_len, features_len))
    for i in range(X.shape[1]):
        for j in range(5):    
            means[j, i] = class_sum[j][i]/lengths[j]

    #sub groups - global mean (mean corrected data):
    globalMean= [np.mean(X[:, 0]), np.mean(X[:, 1])]

    n = []
    c_k = []
    for i in range(class_len):
        n.append(len(indices[i])/len(X))
        mean_corrected_data = np.empty((len(indices[i]), features_len))
        for index, j in enumerate(indices[i]):
            mean_corrected_data[index] = X[j]-globalMean
        c_k.append(np.matmul(np.transpose(mean_corrected_data), mean_corrected_data)/ len(mean_corrected_data))

    covmat = np.zeros((features_len, features_len))
    for i in range(0, features_len):
        for j  in range(0, features_len):
            for k in range(class_len):
                covmat[i,j] += n[k]*c_k[k][i,j]
                
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    class_array = np.unique(y)
    features_len = X.shape[1]
    class_len = len(class_array)

    indices = []
    for i in range(class_len):
        indices.append(np.where(y==class_array[i])[0])

    class_sum = []
    for i in indices:
        total = 0;
        for j in i:
            total += X[j]
        class_sum.append(total)

    lengths = []
    for i in indices:
        lengths.append(len(i))


    means = np.ones((class_len, features_len))
    for i in range(X.shape[1]):
        for j in range(5):    
            means[j, i] = class_sum[j][i]/lengths[j]

    covmats = []
    for i in range(class_len):
        mean_corrected_data = np.empty((len(indices[i]), features_len))
        for index, j in enumerate(indices[i]):
            mean_corrected_data[index] = X[j]- means[i]
        covmats.append(np.matmul(np.transpose(mean_corrected_data), mean_corrected_data)/ len(mean_corrected_data))
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    f = np.zeros((len(means), len(Xtest)))
    for i in range(len(means)):
        f[i] = np.matmul(means[i], np.matmul(inv(covmat),np.transpose(Xtest)))- 0.5*np.matmul(means[i], np.matmul(inv(covmat), np.transpose(means[i])))
    ypred = np.argmax(f, axis = 0) + np.ones((len(Xtest),))
    
    #accuracy formula 
    true_true = 0.0;
    for i in range(len(ypred)):
        if(ypred[i] == ytest[i]):
            true_true += 1
    acc = true_true/len(ypred) * 100
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    p = np.zeros((len(means), len(Xtest)))
    for i in range(len(Xtest)):
        for j in range(len(means)):
            p[j,i] = 1/(2*pi*sqrt(det(covmats[j]))) * np.exp((-0.5)* np.matmul(np.matmul(Xtest[i]-means[j], inv(covmats[j])), np.transpose(Xtest[i]-means[j])))
    ypred = np.argmax(p, axis = 0) + np.ones((len(Xtest),))
    
    true_true = 0.0;
    for i in range(len(ypred)):
        if(ypred[i] == ytest[i]):
            true_true += 1
    acc = true_true/len(ypred) * 100
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    w = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))                                                      
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    w = np.matmul(inv(np.matmul(X.T, X) + np.multiply(lambd, np.identity(X.shape[1]))), np.matmul(X.T, y))                                                  
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    mse = 1/len(Xtest) * np.dot((ytest - np.dot(Xtest, w)).T, (ytest - np.dot(Xtest, w)))
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    x_size = X.shape[1]
    error = np.matmul((y - np.matmul(X, w.reshape(x_size, 1))).T, (y - np.matmul(X, w.reshape(x_size, 1)))) + lambd * np.matmul(w.reshape(x_size,1).T, w.reshape(x_size,1))
    summation = 0
    error_grad = 0
    for i in range(len(y)):
        summation = summation + ((y[i]- np.matmul(w.T, X[i]))*X[i])
    error_grad = -(summation) + np.dot(lambd, w)
    
    error = np.squeeze(error)
    error_grad = np.squeeze(error_grad)
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    Xp = np.ones((len(x), p+1))
    for i in range(len(x)):
        for j in range(p+1):
            Xp[i,j] = x[i]**j
    return Xp
# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.05
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
