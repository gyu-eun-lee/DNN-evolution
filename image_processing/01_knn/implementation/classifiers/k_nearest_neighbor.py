from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        dists = self.compute_distances_no_loops(X)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        """
        Optimization: uses broadcasting and numpy vectorized operations to
        compute efficiently.
        Also using dist(x,y)**2 == x**2 - 2x*y + y**2 to take advantage of
        efficient matrix multiplications.
        """
        dists = np.sqrt(np.sum(X**2, axis=1, keepdims=True) - 2*np.dot(X,self.X_train.T) + np.sum(self.X_train**2, axis=1))

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            """
            A list of length k storing the labels of the k nearest neighbors to the ith test point.
            """
            closest_y = []
            """
            Use distance matrix to kind k nearest neighbors of ith testing
            point, use self.y_train to find labels of neighbors, store
            labels in closest_y.
            """
            for n in np.argsort(dists[i])[0:k]:
              closest_y.append(self.y_train[n])

            """
            Predicted label = most common label in closest_y. Store as
            y_pred[i], break ties using smaller label.
            """
            values, counts = np.unique(closest_y, return_counts = True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred
