from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Not used in notebook, mainly here to provide a more readable description
    of vectorized loss function

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                """
                (with transposes on X since f(x[i],W) = W*x.T)
                dW[k][l] = (1/N)*sum([(dL/ds_j)(X[i])*(ds_j/dW_kl)(X[i]) 
                      for i in range(num_train), j in range(num_classes)])

                (dL/ds_j)(X[i]) = {1 if margin > 0, 0 else} (y != j)
                (dL/ds_y)(X[i]) = {-1 if margin > 0, 0 else}

                (ds_j/dW_kl)(X[i]) = dirac(j,k)*X[i][l]

                j != y[i] and margin > 0 -> dW[j][l] += X[i][l] (j != y)
                                         -> dW[y][l] += -X[i][l] (y != j)
                """
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Convert loss from sum over training data into an average
    loss /= num_train
    dW /= num_train
    # Add regularization term
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    correct_class_score = np.array([[scores[i][y[i]]]*num_classes for i in range(num_train)])
    margin = scores - correct_class_score + (scores - correct_class_score != 0)
    loss = np.sum(np.maximum(np.zeros(margin.shape),margin)) / num_train + reg * np.sum(W * W)
    # margin > 0 also picks out j != y
    dW += (margin > 0).T.dot(X).T
    X_count = np.multiply(X.T, (margin > 0).sum(axis=1)).T
    # margin == 0 picks out j == y
    dW -= (margin == 0).T.dot(X_count).T
    dW /= num_train
    # regularization term
    dW += 2 * reg * W

    return loss, dW