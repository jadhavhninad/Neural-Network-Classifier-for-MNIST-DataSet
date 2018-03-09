### [](#header-3) Goals:
To implement a two layer neural network for a binary classifier and a multi layer neural network for a multiclass classifier. Compare performance with K-Nearest Neighbour approach for same dataset size.

### [](#header-3) Implementation:
*   The two layer network has 1 hidden layer dimension=500, for binary classification. The multilayer neural network program is able to create and train a multilayer network based on command line arguments. 
*   The binary NN classifier has been modularized so that it can now be extended to multiclass classification by replacing the final sigmoid function with softmax
*   Two approaches for K-Nearest Neighbour have been tried - using the default python package and building the algorithm from scratch. 
*   Since the MNIST dataset has 60,000 images which is too large for batch gradient descent. Therefore, training is done with 6000 samples and test with 1000 samples.

### [](#header-3) Output:
*   The training and testing accuracies. 
*   Plot of train error vs iterations
*   Plot of K-value vs Error (for kNN)


#Performance comparison using KNN.
Max accuracy of ~94% was achieved using KNN approach. The default KNN package was used as well as an algorithm was built from scratch.

