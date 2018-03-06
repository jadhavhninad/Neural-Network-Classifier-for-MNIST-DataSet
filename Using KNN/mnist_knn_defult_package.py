'''
Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
Author : Ninad Jadhav(Id : 1213245837)
'''

from sklearn.datasets import fetch_mldata
import numpy as np
from  matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def euclidean_distance(a,b):
    distance = np.sum(np.power((np.subtract(a,b)),2))
    return distance

def knn_model(X,y,test_X,k):
    pred_y=np.array(np.zeros(test_X.shape[0]))
    for test_sample in range (0,test_X.shape[0],1):
        vote=[]
        for train_sample in range (0,X.shape[0],1):
            dist = euclidean_distance(X[train_sample,:],test_X[test_sample,:])
            vote.append((dist, y[train_sample]))

        k_nearest = heapq.nsmallest(k,vote,key=lambda x:x[0])
        digit_pred={0.0:0, 1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0}

        for neighbours,label in k_nearest:
            digit_pred[label] += 1

        pred_y[test_sample] = max(digit_pred.items(), key=operator.itemgetter(1))[0]
        
    return pred_y


def main():
    mnist = fetch_mldata('MNIST original', data_home="./mnist_dataset")
    '''
    print(mnist.data.shape)
    print(mnist.target.shape)
    '''
    original_dataX = np.array(mnist.data)
    original_dataY = np.array(mnist.target.reshape((mnist.target.shape[0],1)))
    data = np.concatenate((original_dataX,original_dataY),axis=1)
    np.random.shuffle(data)
    print(data[69990:70000,data.shape[1]-1])
    
    X = np.array(data[0:6000, 0:data.shape[1]-1])
    y = np.array(data[0:6000, data.shape[1]-1])

    test_X = np.array(data[69000:70000, 0:data.shape[1]-1])
    test_y = np.array(data[69000:70000, data.shape[1]-1])

    '''
    print (np.unique(y))  
    print(np.unique(test_y))

   
    print(X[0,:])
    print(y[0])
    print(test_X[0,])
    print(test_y[0])
    '''
    
    accuracy={}
    kvals = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    #kvals = [1]
    for k in kvals:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, y)
        prediction_y = neigh.predict(test_X)
        accuracy[k] = np.sum(((test_y == prediction_y)/test_y.shape[0])) * 100
        print(k, accuracy[k])
    
if __name__ == "__main__":
    main()
