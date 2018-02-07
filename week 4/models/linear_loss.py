import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_pred = [sum([k*p for k, p in zip(x, W)]) for x in X]
    for _y, _y_real in zip(y_pred, y):
        loss += (_y - _y_real) ** 2
    loss /= 2 * len(y_pred)
    dW = sum([_y - _y_real]*x for x, _y, _y_real in zip(X, y_pred, y)) / len(y_pred)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_pred = X.dot(W)
    loss_array = y_pred - y
    loss = sum(loss_array * loss_array) / (2 * len(y_pred))
    temp = np.array(y_pred - y)
    dW = (X.transpose().dot(temp)) / len(y_pred)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW