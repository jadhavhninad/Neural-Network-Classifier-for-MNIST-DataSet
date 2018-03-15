'''
Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
Author : Ninad Jadhav(Id : 1213245837)
'''

from sklearn.datasets import fetch_mldata
from numpy import genfromtxt
import numpy as np
from  matplotlib import pyplot as plt
import heapq
import operator

'''
Calculate the euclidean distance. Using Euclidean square distance will have no impact on the
final classification
'''
def euclidean_distance(a,b):
    #distance = np.power(np.sum(np.power((np.subtract(a,b)),2)),0.5)
    distance = np.sum(np.power((np.subtract(a,b)),2))
    return distance

'''
Calculate KNN neighbours for each test data point for the max value of K and return them to the
used in the precompute KNN method
'''
def knn_model(X,y,test_X,maxK):
    sample_Kstat=[]
    for test_sample in range (0,test_X.shape[0],1):
        vote=[]
        for train_sample in range (0,X.shape[0],1):
            dist = euclidean_distance(X[train_sample,:],test_X[test_sample,:])
            vote.append((dist, y[train_sample]))

        k_nearest = heapq.nsmallest(maxK,vote,key=lambda x:x[0])
        sample_Kstat.append(k_nearest)

    return sample_Kstat

'''
Using the precomputed list of neighbours for every k value use first K nearest points to classify
'''
def knn_precompute(X,y,test_X,k,Kstat):
    pred_y=np.array(np.zeros(test_X.shape[0]))
    for test_sample in range (0,test_X.shape[0],1):
        k_nearest = Kstat[test_sample][0:k]
        digit_pred={0.0:0, 1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0}
        for neighbours,label in k_nearest:
            digit_pred[label] += 1

        pred_y[test_sample] = max(digit_pred.items(), key=operator.itemgetter(1))[0]
        
    return pred_y


def main():
    #Set update to false if there already exists a file knn_data.csv. Set to true while testing new data
    update = True;
    '''
    mnist = fetch_mldata('MNIST original', data_home="./mnist_dataset")
    
    print(mnist.data.shape)
    print(mnist.target.shape)
    
    original_dataX = np.array(mnist.data)
    original_dataY = np.array(mnist.target.reshape((mnist.target.shape[0],1)))
    data = np.concatenate((original_dataX,original_dataY),axis=1)
    np.random.shuffle(data)
    #print(data[69990:70000,data.shape[1]-1])

    '''
    data_train = genfromtxt('mnist_train.csv', delimiter=',', max_rows=6000)
    #np.random.shuffle(data_train)
    data_test = genfromtxt('mnist_test.csv', delimiter=',', max_rows=1000)
    
    X = np.array(data_train[:, 1:data_train.shape[1]])
    y = np.array(data_train[:, 0])

    test_X = np.array(data_test[0:1000, 1:data_test.shape[1]])
    test_y = np.array(data_test[0:1000, 0])
    '''
    
    X = np.array(data[0:6000, 0:data.shape[1]-1])
    y = np.array(data[0:6000, data.shape[1]-1])

    test_X = np.array(data[60000:61000, 0:data.shape[1]-1])
    test_y = np.array(data[60000:61000, data.shape[1]-1])

    print(X.shape)

    print(X[0,:])
    print(y[0])
    print(test_X[0,])
    print(test_y[0])
    '''
    accuracy=[]
    kvals = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    maxK_val = max(kvals)
    MaxKNN = knn_model(X,y,test_X,maxK_val)

        
    for k in kvals:
        prediction_y = knn_precompute(X,y,test_X,k,MaxKNN)
        acc = np.sum(((test_y == prediction_y)/test_y.shape[0])) * 100
        print(k, acc)
        accuracy.append(acc)

    kv = np.asarray(kvals)
    print("For k = ", k , "Accuracy % is = ", np.asarray(accuracy))
    plt.plot(np.asarray(kvals),np.asarray(accuracy))

    # labels
    plt.title('Test Error Plot for K neighbour')
    plt.xlabel('K value')
    plt.ylabel('Test Error')

    plt.savefig("Test Error Plot for K neighbour")
    plt.show()
    
if __name__ == "__main__":
    main()
