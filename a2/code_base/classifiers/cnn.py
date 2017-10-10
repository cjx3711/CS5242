from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    

    F = num_filters
    # Conv layer
    # Input: (N, C, H, W)
    # Output: (N, F, Hc, Wc)
    # ConvW: (F, C, HH, WW)
    C, H, W = input_dim
    w1Shape = (F, C, filter_size, filter_size) 
    w1 = np.random.randn(np.product(w1Shape)).reshape(w1Shape) * weight_scale
    b1 = np.zeros((F,))
    print(w1.shape)
    print(b1.shape)
    pad = (filter_size - 1)
    Hc = 1 + (H + pad - filter_size) # / stride (assume stride = 1)
    Wc = 1 + (W + pad - filter_size) # / stride (assume stride = 1)
    
    # Pool layer  (No weights)
    # Input: (N, F, Hc, Wc)
    # Output: (N, F, Hp, Wp)
    Hp = 1 + (Hc - 2) // 2
    Wp = 1 + (Wc - 2) // 2
    
    # Hidden Affine layer
    # Input: (N, F * Hp * Wp)
    # Output: (N, Ha)
    # HidAffineW: (F * Hp * Wp, Ha)
    w2 = np.random.randn(F * Hp * Wp, hidden_dim)
    b2 = np.zeros((hidden_dim,))
    
    # Output Affine layer
    # Input: (N, Ha)
    # Output: (N, NC)
    # AffineW: (Ha, NC)
    w3 = np.random.randn(hidden_dim, num_classes)
    b3 = np.zeros((num_classes,))
    
    self.params = {
        "W1": w1, "b1": b1,
        "W2": w2, "b2": b2,
        "W3": w3, "b3": b3
    }
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N = X.shape[0]
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    N, F, Hp, Wp = out1.shape
    out1 = out1.reshape((N, F * Hp * Wp))
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)
    
    # print("out1.shape {0}".format(out1.shape))
    # print("out2.shape {0}".format(out2.shape))
    # print("scores.shape {0}".format(scores.shape))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    
    loss, grads = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(W1**2)
    loss += 0.5 * self.reg * np.sum(W2**2)
    loss += 0.5 * self.reg * np.sum(W3**2)
    
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    dh3, dw3, db3 = affine_backward(grads, cache3)
    dh2, dw2, db2 = affine_relu_backward(dh3, cache2)
    dh2 = dh2.reshape(N, F, Hp, Wp)
    dh1, dw1, db1 = conv_relu_pool_backward(dh2, cache1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    grads = {
        "W1": dw1 + self.reg * W1, "b1": db1,
        "W2": dw2 + self.reg * W2, "b2": db2,
        "W3": dw3 + self.reg * W3, "b3": db3
    }
    return loss, grads
  
  
pass
