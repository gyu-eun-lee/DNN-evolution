from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        # loss = -(correct score) + LogSumExp(score of label j)
        loss -= scores[y[i]]
        loss += np.log(np.sum(np.exp(scores)))
        for c in range(num_classes):
            if c == y[i]:
                dW[:,c] -= X[i]
            dW[:,c] += (np.exp(scores[c])/np.sum(np.exp(scores)))*X[i]

    loss /= num_train
    dW /= num_train
    loss += reg*np.sum(W*W)
    dW += 2*reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    loss -= np.sum(scores[y])
    loss += np.sum(np.log(np.sum(np.exp(scores), axis=1)))
    cEqualsy = np.array([c-y == 0 for c in range(num_classes)])
    dW -= X.T.dot(cEqualsy.T)
    dW += (np.reciprocal(np.sum(np.exp(scores),axis=1))*X.T).dot(np.exp(scores))
    loss /= num_train
    dW /= num_train
    loss += reg*np.sum(W*W)
    dW += 2*reg*W

    return loss, dW
