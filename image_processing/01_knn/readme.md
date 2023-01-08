# $k$-nearest neighbor classifier

This miniature model implements a very crude single-layer model for image classification on the CIFAR-10 dataset. It uses the $k$-nearest neighbor (KNN) classifier with the $L^2$ norm, with hyperparameter optimization for the number of neighbors $k$.

## Model overview

* Input: Image $X$, a 2-d array of pixel values reshaped into an $N$-dimensional vector.
* Target: Predicted label $y\in \text{labels}$, where $\text{labels}$ is a set of $C$ classification labels for each image (e.g. "cat", "boat", etc).
* Model parameters: 
    * $M = \{(X_t,y_t)\}$: all image-label pairs $(X_t,y_t)$ in the training dataset.
    * Hyperparameters:
        * $k$: number of neighbors to use in the $k$-nearest neighbors calculation

**Prediction generation algorithm:**

1. Take image $X$ as input.
2. For each $(X_t,y_t)\in M$, compute $d(X,X_t)$ where $d(U,V) = \|U-V\|_2$ is the $L^2$ distance between vectors $U$ and $V$. Store the results $\text{dists}(X;M) = \{(X_t,d(X,X_t))\}_{X_t\in M}$.
3. Find training images $(X_t^{(1)},\ldots,X_t^{(k)})$ corresponding to the $k$ smallest values of $d(X,X_t)$ in $\text{dists}(X;M)$. These are the $k$ nearest neighbors of $X$ in the training dataset. Find their corresponding labels $(y_t^{(1)},\ldots,y_t^{(k)})\in\text{labels}^k$.
4. Generate prediction $y$ by taking the most common label among the $k$ nearest neighbors, i.e. $y = \text{mode}(y_t^{(1)},\ldots,y_t^{(k)})$.


## Training

Training the model simply consists of memorizing all of the training data, so that $M = \{(X_t,y_t)\}$ consists of all image-label pairs $(X_t,y_t)$ in the training dataset.

## Hyperparameter tuning:

The model takes a single hyperparameter $k$, the number of nearest neighbors to consider. Optimization for this hyperparameter is performed via $n$-fold cross-validation (in this particular case $n=5$).

## Performance:

With $k = 10$, the model predicts labels in the test dataset with ~28% accuracy.