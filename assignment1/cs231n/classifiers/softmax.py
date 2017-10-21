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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  for i in range(N):
      # X[i] * W
      class_scores = np.dot(X[i], W)
    
      # shift volume, make numerically stable
      class_scores -= np.max(class_scores)
    
      # exp
      exp_class_scores = np.exp(class_scores)
    
      # compute denominator sum
      denominator = np.sum(exp_class_scores)
      
      # Compute gradient for each sample, each class
      for c in range(C):
          prob = exp_class_scores[c] / denominator
          if y[i] == c:
              dW[:, c] += (prob - 1.) * X[i]
          else:
              dW[:, c] += prob * X[i]
      
      # Compute loss
      loss_per_example = exp_class_scores[y[i]] / denominator
      loss += -np.log(loss_per_example)
    
  # Average loss and account for regularization term
  loss /= N
  loss += reg * np.sum(W * W)
  
  # average gradient
  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

