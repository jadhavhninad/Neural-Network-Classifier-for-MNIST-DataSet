'''
This file implements a two layer neural network for a binary classifier
'''
import numpy as np
from load_dataset import mnist
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    return A

def tanh_der(dA, Z):
    '''
	computes derivative of tanh activation

	Inputs: 
		dA is the derivative from subsequent layer. numpy.ndarray (n, m)
		cache is a dictionary with {"Z", Z}, where Z was the input 
		to the activation layer during forward propagation

	Returns: 
		dZ is the derivative. numpy.ndarray (n,m)
	'''
    ### CODE HERE
    A = tanh(Z)
    dZ = dA * (1 - A**2)
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    return A

def sigmoid_der(dA, Z):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    
    Z = cache["Z"]
    '''
    
    A = sigmoid(Z)
    dZ = dA * A*(1 - A)
    ### CODE HERE
    return dZ

#def initialize_2layer_weights(n_in, n_h, n_fin):
def initialize_2layer_weights(net_dims):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE

    parameters=[]
    for i in range(0,len(net_dims)-1,1):
        param=[]
        W = np.random.rand(net_dims[i+1],net_dims[i])* 0.01
        b = np.random.rand(net_dims[i+1],1) * 0.1
        param.append(W)
        param.append(b)
        parameters.append(param)
        
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    Z = np.dot(W,A) + b
    cache = {}
    cache["A"] = A
    cache["W"] = W
    cache["b"] = b
    cache["Z"] = Z
    
    return cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(cache["Z"])
    elif activation == "tanh":
        A = tanh(cache["Z"])

    return A, cache

def cost_estimate(A_last, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A_last - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function
    '''
    m=Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(A_last) + (1-Y)*np.log(1-A_last))
	
    ### CODE HERE
    return cost

def linear_backward(dZ, A, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A_prev
            where Z = W_curretnA_Prev + b_current,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    dW = np.dot(dZ,A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
	
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the A_prev, W, b, Z values
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    
    if activation == "sigmoid":
        dZ = sigmoid_der(dA, cache["Z"])
    elif activation == "tanh":
        dZ = tanh_der(dA, cache["Z"])
   
    dA_prev, dW, db = linear_backward(dZ, cache["A"], W, b)
    return dA_prev, dW, db

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    W1,b1 = parameters[0]
    W2,b2 = parameters[1]
    A1, cache1 = layer_forward(X, W1, b1, "tanh")
    A2, cache2 = layer_forward(A1, W2, b2, "sigmoid")

    YPred = np.zeros((1, X.shape[1]))

    for i in range(A2.shape[1]):
        if A2[0, i] >= 0.5:
            YPred[0, i] = 1
        else:
            YPred[0, i] = 0

    return YPred

def two_layer_network(X, Y, net_dims, num_iterations=2000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent

    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters = initialize_2layer_weights(net_dims)

    W=[]
    b=[]
    dW=[]
    db=[]
    costs = []

    #dummy initialization of dW and db
    for k in range(len(parameters)):
        tw,tb = parameters[k]
        W.append(tw)
        b.append(tb)
        dW.append(tw)
        db.append(tb)

    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A_prev = X
        A_vals=[]
        cache_vals=[]
        for itr in range(len(parameters)-1):
            tempA,temp_cache = layer_forward(A_prev, W[itr], b[itr], "tanh")#For all the middle hidden layers
            A_vals.append(tempA)
            cache_vals.append(temp_cache)
            A_prev = tempA
            #print("done with itr ",itr)

        A_final, cache_final = layer_forward(A_prev, W[len(W)-1], b[len(b)-1], "sigmoid")#For last layers since binary clssification
        cache_vals.append(cache_final)

        # A_final = Prediction (binary classifier)
        # cost estimation
        ### CODE HERE
        cost = cost_estimate(A_final, Y)
        m_s = Y.shape[1]

        #The matrix operations are done element-wise
        dA = (-1/m_s) * (Y/A_final - (Y - 1) / (A_final - 1))		

        
        for itr in range(len(parameters)-1, -1, -1):
            dA_prev, dW[itr], db[itr] = layer_backward(dA, cache_vals[itr], W[itr], b[itr], "sigmoid")
            dA = dA_prev

        # update parameters
        ### CODE HERE
        for itr in range(len(parameters)):
            W[itr] += -learning_rate * dW[itr]
            b[itr] += -learning_rate * db[itr]

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" % (ii, cost))

    return costs, parameters


def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 2 and 3
    # for train, from the first 1000 samples, we get those samples which have label 2,4
    # for train, from the first 6000 samples, we get those samples which have label 2,4
    train_data, train_label, test_data, test_label = \
                mnist(ntrain=6000,ntest=1000,digit_range=[2,4])

    #the labels are converted to 0 and 1 values (instead of the actual numbers)
    
    n_in, m = train_data.shape
    n_fin = 1
    n_h = 500
    net_dims = [n_in, n_h, n_fin]
    
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 1000

    costs, parameters = two_layer_network(train_data, train_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    trAcc = 100 - (np.sum((np.abs(train_Pred - train_label)))/train_Pred.shape[1])*100
    teAcc = 100 - (np.sum((np.abs(test_Pred - test_label)))/train_Pred.shape[1])*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    # CODE HERE TO PLOT costs vs iterations

    plt.plot(costs)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.savefig('error_plot.png')


if __name__ == "__main__":
    main()




