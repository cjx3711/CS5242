from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x >= 0) * dout
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction.
         We will use the same definition in lecture notes 3b, slide 13 (ie. same padding on both sides).

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + pad - HH) / stride
      W' = 1 + (W + pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # print("x.shape {0}".format(x.shape))
    # print("pX.shape {0}".format(pX.shape))
    # print("w.shape {0}".format(w.shape))
    # 
    # print("b.shape {0}".format(b.shape))
    # print("conv_param {0}".format(conv_param))
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    pX = np.pad(x, [(0,0), (0,0), (pad//2,pad//2), (pad//2,pad//2)], mode='constant') # Pad zeros only on the H and W axis
    oH = int(1 + (H + pad - HH) / stride)
    oW = int(1 + (W + pad - WW) / stride)
    out = np.zeros((N, F, oH, oW))
    # print("out.shape {0}".format(out.shape))
    
    for hI in range(oH):
        fromH = hI * stride
        tillH = fromH + HH
        for wI in range(oW):
            fromW = wI * stride
            tillW = fromW + WW
            
            # print ("fromH, tillH, fromW, tillW: {0}, {1}, {2}, {3}".format(fromH, tillH, fromW, tillW))
            xPart = pX[:,:,fromH:tillH, fromW:tillW]
            wPart = w[:,:,:,:]
            # print(xPart.shape)
            # print(wPart.shape)

            xFlat = xPart.reshape( N, C * HH * WW )
            wFlat = wPart.reshape( F, C * HH * WW )

            # print(xFlat.shape)
            # print(wFlat.shape)
            result = np.dot(xFlat, wFlat.T) + b
            # print(out[:, :, hI, wI].shape)
            # print(result.shape)
            out[:, :, hI, wI] = result
            # print(result)
    # print(out)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    
    - x: Input data shape (N, C, H, W)
    - w: Filter weights shape (F, C, HH, WW)
    - b: Biases shape (F,)
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # print("dout.shape {0}".format(dout.shape))
    # dx, dw, db = np.zeros((4,3,5,5)), np.zeros((2,3,3,3)), np.zeros((2,))
    x, w, b, conv_param = cache
    N, C, H, W = x.shape 
    F, C, HH, WW = w.shape
    _, _, DH, DW = dout.shape
    
    dx, dw = np.zeros(x.shape), np.zeros(w.shape)
    print("dx.shape {0}".format(dx.shape))
    print("dw.shape {0}".format(dw.shape))
    
    stride, pad = conv_param['stride'], conv_param['pad']
    padH = pad // 2
    oH = int(1 + (H + pad - HH) / stride)
    oW = int(1 + (W + pad - WW) / stride)

    kernelSize = C * HH * WW
    pDX = np.zeros((N, C, H + pad, W + pad))
    conv = np.reshape(w, (F, kernelSize))
    
    print("pDX.shape {0}".format(pDX.shape))
    print("conv.shape {0}".format(conv.shape))
    
    pX = np.pad(x, ((0, 0), (0, 0), (padH, padH), (padH, padH)), mode='constant') # Pad zeros only on the H and W axis
        
    dwReshaped = np.zeros((F, kernelSize))
    for hI in range(oH):
        fromH = hI * stride
        tillH = fromH + HH
        for wI in range(oW):
            fromW = wI * stride
            tillW = fromW + WW

            doutPart = dout[:, :, hI, wI]

            dxSub = doutPart.dot(conv)
            dxSubReshaped = dxSub.reshape(N, C, HH, WW)
            pDX[:, :, fromH:tillH, fromW:tillW] += dxSubReshaped

            # x_sub has dim (N, C*HH*WW),
            # doutPart has dimension (N, F)
            # dwReshaped has dim (F, C*HH*WW)
            xSubReshaped = pX[:, :, fromH:tillH, fromW:tillW].reshape(N, C * HH * WW)
            dwReshaped += doutPart.T.dot(xSubReshaped)

    dx = pDX[:, :, padH:-padH, padH:-padH]
    dw = dwReshaped.reshape((F, C, HH, WW))
    db = dout.sum(axis=(0, 2, 3)) # sum along axis N, H', and W'
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    Forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    oW = int((W - pool_width) / stride) + 1
    oH = int((H - pool_height) / stride) + 1

    out = np.zeros((N, C, oH, oW))
    
    for hI in range(oH):
        fromH = hI * stride
        tillH = fromH + pool_height
        for wI in range(oW):
            fromW = wI * stride
            tillW = fromW + pool_width

            out[:, :, hI, wI] = x[:, :, fromH:tillH, fromW:tillW].max(axis=(2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    oW = int((W - pool_width) / stride) + 1
    oH = int((H - pool_height) / stride) + 1

    dx = np.zeros((N, C, H, W))
    
    for hI in range(oH):
        fromH = hI * stride
        tillH = fromH + pool_height
        for wI in range(oW):
            fromW = wI * stride
            tillW = fromW + pool_width

            doutCol = dout[:, :, hI, wI].reshape(N*C)
            viewCol = x[:, :, fromH:tillH, fromW:tillW].reshape((N * C, pool_height*pool_width))
            dxViewCol = dx[:, :, fromH:tillH, fromW:tillW].reshape((N * C, pool_height*pool_width)).T

            pos = np.argmax(viewCol, axis=1)

            dxViewCol[pos, range(N * C)] += doutCol

            dxView = dxViewCol.T.reshape(N, C, pool_height, pool_width)
            dx[:, :, fromH:tillH, fromW:tillW] += dxView
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
