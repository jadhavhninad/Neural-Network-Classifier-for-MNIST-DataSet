# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:15:18 2018

@author: nhjadhav
python version:3.6

"""

import numpy as np
from load_dataset import mnist
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
import sys, ast

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    
  
    #print(Z.shape)
    #print(dZ.shape)
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def initialize_multilayer_weights(m,net_dims,filtersz=3,convLcount=1,channel=1,mp_layers=1):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    
    parameters = {}
    P=0 #padding
    S=1 #stride
    
    #all next channels are 5
    curr_ch = 1
    next_ch = channel
    wd_size = net_dims[0]
    #parameters for convLayers
    for l in range(convLcount):
        parameters["cW"+str(l+1)] = np.random.rand(filtersz,filtersz,curr_ch,next_ch)*0.01
        parameters["cb"+str(l+1)] = np.random.rand(1,1,1,next_ch)*0.01
        wd_size = wd_size -filtersz +1
  
    #print(wd_size)
    mp_filter=2
    for l in range(mp_layers):
        wd_size -= mp_filter-1
        
    #parameters for fully connected layers
    net_dims[0] = wd_size * wd_size * next_ch
    #print(wd_size)
    full_layers = len(net_dims)
    for l in range(full_layers-1):
        parameters["W"+str(l+1)] = np.random.rand(net_dims[l+1],net_dims[l])*0.01
        parameters["b"+str(l+1)] = np.random.rand(net_dims[l+1],1)*0.01
    return parameters


def linear_forward_conv(A, W, b, ch, filtersz=3):

    samples = A.shape[0]
    limit =  A.shape[1] - filtersz + 1

    Z = np.random.rand(samples,limit,limit,ch)
    for m in range(A.shape[0]):
        for start in range(limit):
                for end in range(limit):
                    for channel in range(ch):
                        Z[m,start,end,channel] = np.squeeze(np.sum(np.multiply(A[m,start:start+filtersz, end:end+filtersz,0],W[:,:,0,channel])) + b[:,:,0,channel])

    cache = {}
    cache["A"] = A
    return Z, cache

def linear_forward(A, W, b):
    
    Z = np.dot(W,A) + b

    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b, ch, activation,layerType):

    if layerType == 'conv':
        Z, lin_cache = linear_forward_conv(A_prev, W, b,ch,3)     
    else:
        Z, lin_cache = linear_forward(A_prev, W, b)
    
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache


def max_pool(A):
    samples = A.shape[0]
    filtersz = 2
    limit =  A.shape[1] - filtersz + 1
    ch = 5
    Z = np.random.rand(samples,limit,limit,ch)
    for m in range(A.shape[0]):
        for start in range(limit):
                for end in range(limit):
                    for channel in range(ch):
                        Z[m,start,end,channel] = np.squeeze(np.max(A[m,start:start+filtersz, end:end+filtersz,0]))
    
    return Z

def max_pool_back(Z,mp_cache):
    dA_prev = mp_cache["maxpool"]
    filtersz = 2
    ch=5
    limit =  dA_prev.shape[1] - filtersz + 1
                    
    for m in range(dA_prev.shape[0]):
        for start in range(limit):
            for end in range(limit):
                for channel in range(ch):
                    pos=np.argmax(dA_prev[m,start:start+filtersz, end:end+filtersz,0], axis=0)
                    dA_prev[pos] += Z[m,start,end,0]
    
    return dA_prev

def layer_forward_do(A_prev):
    P = np.random.rand(A_prev.shape[0], A_prev.shape[1])
    P = P < 0.5
    A = np.multiply(A_prev, P)
    A = A/0.5
    cache={}
    cache["P"] = P
    cache["p"] = 0.5
    return A,cache
    

def multi_layer_forward(X, parameters,ch, convLcount=1,do="no"):

    L = len(parameters)//2 - convLcount  
    A = X
    caches = []
    
    #for conv layers
    for l in range(1,1+convLcount):
        #print("param cW Shape = ", parameters["cW"+str(l)].shape)
        #print("param cb Shape = ", parameters["cb"+str(l)].shape)
        A, cache = layer_forward(A, parameters["cW"+str(l)], parameters["cb"+str(l)], ch ,  "relu", "conv")
        caches.append(cache)
    
    mp_cache={}
    mp_cache["maxpool"] = A
    #print("mp in ml fwd = ", mp_cache["maxpool"].shape)
    A = max_pool(A)
    
    A = A.reshape(A.shape[0],A.shape[1] * A.shape[2] * A.shape[3])
        
    A = A.T
    #print(A.shape)
    #print("Going to FC layer")
    #For fully connected layer
    #print("A going to FC layer shape ", A.shape)
    drp_ot_cache=[]
    if do=="yes":
        A, do_cache = layer_forward_do(A)
        drp_ot_cache.append(do_cache)
    for l in range(1,L):
        #print("param W Shape = ", parameters["W"+str(l)].shape)
        #print("param b Shape = ", parameters["b"+str(l)].shape)

        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], 0, "relu", "full")
        if do=="yes":
            A, do_cache = layer_forward_do(A)
            drp_ot_cache.append(do_cache)
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], 0, "linear", "full")
    caches.append(cache)
    return AL, caches, mp_cache, drp_ot_cache

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE 
    loss = 0
    A = np.zeros((Z.shape))
    cache={}

    for i in range(0,Z.shape[1],1):
        maxZ = max(Z[:,i])
        C = np.sum(np.exp(Z[:,i] - maxZ))
        #print(C)
        A[:,i] = np.exp(Z[:,i] - maxZ) / C
        #eg: class 3 => Y[i] = 3. So only the activation function of index A[2,i] will contribute to loss.
        if(Y.shape[0]>0):
            loss += (-np.log(A[int(Y[0,i]),i]))
        
    cache["A"] = A
    loss = loss/Z.shape[1]
    
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE 
    dZ = cache["A"]
    for i in range(0,Y.shape[1],1):
        dZ[int(Y[0,i]),i] -= 1

    return dZ

def linear_backward(dZ, cache, W, b, layerType="full"):

    A_prev = cache["A"]
    #print("A shape in lin back", A_prev.shape)
    #print("W shape in lin back", W.shape)
    #print("dZ b4 =" ,dZ.shape)
    
    ## CODE HERE
    if layerType == "conv":
        dA_prev = np.zeros((dZ.shape[0],10,10,1))
        dW = np.zeros((3,3,1,5))
        db = np.zeros((1,1,1,5))
        
        # Looping over vertical(h) and horizontal(w) axis of the output
        for m in range(dZ.shape[0]):
            for h in range(dZ.shape[1]):
                for w in range(dZ.shape[2]):
                    for ch in range(dZ.shape[3]):
                        dA_prev[m,h:h+3,w:w+3,0] += W[:,:,0,ch] * dZ[m,h,w,ch]
                        dW[:,:,0,ch] += A_prev[m,h:h+3,w:w+3,0] * dZ[m,h,w,ch]
                        db[:,:,0,ch] += dZ[m,h,w,ch]
        
    else:
        dA_prev = np.dot(W.T,dZ)
        dW = np.dot(dZ, A_prev.T)
        db = np.sum(dZ,keepdims=True,axis=1)
            
    #print("dZ after ", dZ.shape)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation,layerType="full"):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)


    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b,layerType)
    return dA_prev, dW, db

def reverse_do(dA,cache):
    dA_prev = dA * cache["P"]
    dA_prev = dA/cache["p"]
    return dA_prev

def multi_layer_backward(dAL, caches, parameters,mp_cache, drp_cache, ch, convLcount=1):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    #L = len(caches)  # with one hidden layer, L = 2
    L = len(parameters)//2 - convLcount
    gradients = {}
    dA = dAL
    activation = "linear"
    #print("size of drp_out cache in backprop ", len(drp_cache))
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = layer_backward(dA, caches[l],parameters["W"+str(l)],parameters["b"+str(l)],activation)
        activation = "relu"
        dA = reverse_do(dA,drp_cache[l-1])

    #Update the weights for conv layer
    #print("done FC backprop")
    dA = dA.T
    #print("dA going to max_pool dring backprop ", dA.shape)
    
    #nd = np.power((dA.shape[1]/ch), 0.5)
    #dA = dA.reshape(((dA.shape[0],nd,nd,ch))).astype(float)
    dA = dA.reshape(dA.shape[0],7,7,5)
    #print(len(mp_cache))
    #print("mp in ml bkw = ", mp_cache["maxpool"].shape)
    dA = max_pool_back(dA,mp_cache)
    
    #print(dA.shape)
    for l in reversed(range(1,convLcount+1)):
        dA, gradients["cdW"+str(l)], gradients["cdb"+str(l)] = layer_backward(dA, caches[l-1],parameters["cW"+str(l)],parameters["cb"+str(l)],activation,"conv")
                
    return gradients


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    convLcount = 1
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    #alpha = learning_rate
    L = len(parameters)//2 - convLcount
    ### CODE HERE
    for l in range(1,L+1):
        parameters["W"+str(l)] += -alpha*gradients["dW"+str(l)]
        parameters["b" + str(l)] += -alpha * gradients["db" + str(l)]
    
    for l in reversed(range(1,convLcount+1)):
        parameters["cW"+str(l)] += -alpha*gradients["cdW"+str(l)]
        parameters["cb" + str(l)] += -alpha * gradients["cdb" + str(l)]
        
    #print(alpha)
    return parameters, alpha

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE 
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    ch=5
    AL, caches,mp_cache, do_cache = multi_layer_forward(X, parameters,ch)
    A_SM, sm_cache, cost = softmax_cross_entropy_loss(AL)

    YPred = np.zeros((1, X.shape[0]))

    for i in range(A_SM.shape[1]):
        YPred[0:i] = np.argmax(A_SM,axis=0)

    return YPred

def multi_layer_network(X, y, net_dims,  num_iterations=10, learning_rate=0.1):
    filtersz = 3
    m=X.shape[1]
    ch = 5
    parameters = initialize_multilayer_weights(m,net_dims,filtersz,1,ch)
    epoch=0
    costs = []
    for ii in range(200):
        #print("forward prop")
        AL, caches, mp_cache, do_cache = multi_layer_forward(X, parameters,ch,1,"yes")
        A_SM, sm_cache, cost = softmax_cross_entropy_loss(AL, y)
        
        dZ = softmax_cross_entropy_loss_der(y, sm_cache)
        
        #print("doing backprop")
        #print("in main, passing to backprop function", mp_cache["maxpool"].shape)
        gradients = multi_layer_backward(dZ, caches, parameters,mp_cache, do_cache,ch)
        parameters, alpha = update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.001)
        
        #print("Cost at iteration is: %.05f" %(cost))
        #print("done itr " ,ii)

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
            epoch+=1
        
    return costs, parameters

def main():

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label =  mnist(ntrain=6000,ntest=1000,digit_range=[0,10])
    # initialize learning rate and num_iterations
    init_layer = train_data.shape[1]
    net_dims=[init_layer,500,10]
    costs,parameters = multi_layer_network(train_data, train_label, net_dims, 10, 0.001)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    trAcc = np.sum(train_Pred == train_label)/train_Pred.shape[1] *100
    teAcc = np.sum(test_Pred == test_label)/test_Pred.shape[1] *100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
        ### CODE HERE to plot costs
    plt.plot(costs)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.savefig('error_plot_multiLayer_do_2.png')

if __name__ == "__main__":
    main()