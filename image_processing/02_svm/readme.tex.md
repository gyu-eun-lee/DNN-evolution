# Support vector machine classifier (SVM)

This miniature model implements a simple single-layer model for image classification on the CIFAR-10 dataset. It uses the support vector machine model, using stochastic gradient descent to optimize the loss function.

## **Model overview**

* Input: Image $X$, reshaped into an $D$-dimensional vector.
* Target: Predicted label $y\in \text{labels} = \{1,\ldots,C\}$, where $\text{labels}$ is a set of $C$ classification labels for each image (e.g. "cat", "boat", etc).
* Model parameters:
    * $W\in M_{C\times N}(\mathbb{R})$: a $C\times D$ matrix of weights, regarded as a map $\mathbb{R}^D\to\mathbb{R}^C$.
    * Hyperparameters:
        * $\alpha\in\mathbb{R}_{>0}$: regularization strength. Controls strength of the regularization term $\lambda\|W\|_2^2$.
        * $\epsilon\in\mathbb{R}_{>0}$:  learning rate. Step size for stochastic gradient descent, used during training.
        * $\beta\in\mathbb{N}$: batch size. Sampling size for stochastic gradient descent, used during training.

**Prediction generation algorithm:**

1. Take image $X$ as input.
2. Compute $s(X) = (s(X)_1,\ldots,s(X)_C) = WX \in \mathbb{R}^C$. This vector represents a list of scores for $X$ corresponding to each label in $\text{labels}$. (General SVMs will also have a bias term $b\in\mathbb{R}^C$, so that $s(X) = W\cdot X + b$.)
3. Generate prediction $y$ by taking the label corresponding to the highest score in $s(X)$, i.e. $y = \argmax \{s(X)_i: i=1,\ldots,C\}.$


## **Training**

Training the model consists of generating the matrix of weights $W$ from the training data $\{(X_t,y_t)\}$.

### **1. Construct the loss functional from the training data**

For a given matrix $W$ and a training dataset $\{(X_t,y_t)\}$ (which we regard as fixed), we define the loss functional $\mathcal{L}(W;\{(X_t,y_t)\}) = \mathcal{L}(W)$ as follows:

* For a given image $X$, we set $f(X,W) = s(X,W) = s(X) = WX$ (more generally, $f(X,W) = WX + b$).
* Define the loss functional for the image-label pair $(X,y)$ by the SVM loss function $$L(f(X,W),y) = L(s(X),y) = \sum_{j \neq y} \max(0,s(X)_j - s_y + 1).$$
* Define the loss functional as the average of the loss over the entire training dataset, plus a regularization term: $$\mathcal{L}(W;\{(X_t,y_t)\}) = \frac{1}{N}\sum_{(X_t,y_t)} L(f(X_t,W),y_t) + \alpha R(W)$$
where $N = |\{(X_t,y_t)\}|$, $\alpha$ is the regularization strength, and $R(W) = \|W\|_2^2$ (other functions of $W$ are possible for $R$ too). The regularization helps prevent over-fitting by introducing a data-independent term to the loss.

### **2. Optimize the loss functional in $W$**

Once the loss functional $\mathcal{L}(W;\{(X_t,y_t)\})$ has been constructed from $\{(X_t,y_t)\}$, we regard the training data as fixed and $\mathcal{L}$ as a function of $W$ only: $\mathcal{L} = \mathcal{L}(W)$. We then seek the $W$ that minimizes the average loss, i.e. $$W = \argmin_V \mathcal{L}(V).$$

We accomplish this by gradient descent.

1. Compute the gradient matrix $\nabla_W\mathcal{L}$ by $$(\nabla_W \mathcal{L})_{ij} = \frac{\partial\mathcal{L}}{\partial W_{ij}};$$ it is a matrix of the same dimensions as $W$.
2. Initialize $W^{(0)}$ as a small random matrix, then define iterates $W_n$ by $$W^{(n+1)} = W^{(n)} -\epsilon(\nabla_W\mathcal{L})(W^{(n)})$$
where $\epsilon$ is the learning rate. Continue iterations until $W^{(n)}$ is sufficiently close to the true minimizer of $\mathcal{L}$.

**Sanity checking with FDM:**

The correctness of the analytic (exact) gradient formulas can be verified by implementing a numerical gradient descent using the finite difference method. However, the finite difference method is expensive in practice, so it should only used for verification and not for training the full model.

**Mini-batching:**

For performance, we add an optimization to this procedure called mini-batching (sometimes also called stochastic gradient descent):

1. Instead of computing the loss functional $\mathcal{L}$ using all of the training data $\{(X_t,y_t)\}$, during each iteration of gradient descent we randomly select a subset $B\subset\{(X_t,y_t)\}$ called a *batch*, then compute the loss functional using only training data $(X_t,y_t)$ selected from $B$.
2. We perform all subsequent gradient and gradient descent calculations during this parameter update using this $\mathcal{L}$.

In practice this usually produces a good enough approximation to gradient descent with the full training data, the idea being that $\mathcal{L}$ is an average and a randomly selected sample of the training data should represent it sufficiently well to reproduce $\mathcal{L}$ on average.

## Hyperparameter tuning:

The model takes the hyperparameters $\epsilon$ (learning rate), $\alpha$ (regularization strength), and $\beta$ (batch size). The provided notebook performs hyperparameter tuning in $\epsilon$ and $\alpha$.

## Performance:

With $\epsilon \sim 1.91\cdot 10^{-7}$, $\alpha \sim 2.25\cdot 10^4$, and $\beta = 400$, the model predicts labels in the test dataset with ~37% accuracy.
