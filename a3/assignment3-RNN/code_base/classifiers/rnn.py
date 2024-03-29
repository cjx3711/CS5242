from builtins import range
from builtins import object
import numpy as np

from code_base.layers import *
from code_base.rnn_layers import *
from code_base.data_utils import *

class SentimentAnalysisRNN(object):
    """
    A SentimentAnalysisRNN produces sentiment for a sentennce using a recurrent
    neural network.

    The RNN receives input vectors of size V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the SentimentAnalysisRNN.
    """

    def __init__(self, word_to_idx, hidden_dim=128, fc_dim=128, output_dim=2,
                 max_length=50, cell_type='rnn', dtype=np.float32):
        """
        Construct a new SentimentAnalysisRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - fc_dim: Dimension A for the intermedia output before the final 
          classification layer of the RNN.
        - output_dim: Dimension O for the output sentiment class of the RNN.
        - max_length: Dimension T for the max length of timesteps.
        - cell_type: What type of RNN to use; currently only rnn supported.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        
        # print("Init Model")
        if cell_type not in {'rnn'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.output_dim = output_dim
        self.max_length = max_length
        self.params = {}

        vocab_size = len(word_to_idx)
        
        # Initailization, we remove the randomness for evaluation purpose
        # Initialize parameters for normal RNN
        self.params['Wx'] = np.linspace(-0.2, 0,
                    num=vocab_size*hidden_dim).reshape(vocab_size, hidden_dim)
        self.params['Wx'] /= np.sqrt(vocab_size)
        self.params['Wh'] = np.linspace(0.2, 0.5,
                    num=hidden_dim*hidden_dim).reshape(hidden_dim, hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.linspace(0.2, 0.5, num=hidden_dim)

        # Initialize parameters for temporal affine weights and bias
        self.params['W_a'] = np.linspace(-0.3, 0.3,
                    num=hidden_dim*fc_dim).reshape(hidden_dim, fc_dim)
        self.params['W_a'] /= np.sqrt(2 * hidden_dim)
        self.params['b_a'] = np.linspace(-0.1, -0.3, num=fc_dim)

        # Initialize output to vocab weights
        self.params['W_fc'] = np.linspace(-0.5, 0.5,
                    num=fc_dim*output_dim).reshape(fc_dim, output_dim)
        self.params['W_fc'] /= np.sqrt(fc_dim)
        self.params['b_fc'] = np.linspace(-0.5, 0.5, num=output_dim)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, wordvecs, labels, mask):
        """
        Compute training-time loss for the RNN and update the gradients of the paremters. 
        We input word vectors, ground-truth labels for sentence sentiments and masks for
        the lengths of sentences, and use an RNN to compute loss and gradients on all 
        parameters.

        Inputs:
        - wordvecs: Encoded one-hot word vectors of shape (N, T, V).
        - labels: Ground-truth sentiments; an integer array of shape (N, ) where each 
          element is 0 (positive) or 1 (negative).
        - mask: indices of sentence length, of shape (N, T). Each sentence has a max
          length of T with 0 padding at the end if its actual length is smaller than T.

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """

        # Assume the initial hidden state is a zero vector
        N, T, V = wordvecs.shape
        H = self.params['Wh'].shape[0]
        h0 = np.zeros((N, H))
        
        # print("N: {0}, T: {1}, H: {2}, V: {3}".format(N, T, H, V))

        # Input-to-hidden, hidden-to-hidden, and biases for normal RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the temporal affine layer
        W_a, b_a = self.params['W_a'], self.params['b_a']

        # Weight and bias for the hidden-to-output transformation.
        W_fc, b_fc = self.params['W_fc'], self.params['b_fc']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the SentimentAnaly-  #
        #       -sisRNN.                                                           #
        # In the forward pass you will need to do the following:                   #
        # (1) Use a vanilla RNN to process the sequence of input word vectors      #
        #     and produce an array of hidden states at all timesteps, of shape     #
        #     (N, T, H).                                                           #
        # (2) Use the temporal affine tranformation to map the concatenated        #
        #     array into a new space, producing an array of shape (N, T, A).       #
        # (3) Use an average vector of all timesteps as a hidden representation    #
        #     of each sentence, producing an output array of shape (N, A). Ignore  #
        #     the points where the length is beyond the actual sentence length     #
        #      using the mask above.                                               #   
        # (4) Apply an affine transformation to map it into a 2-D vector of shape  #
        #     (N, 2).                                                              #
        # (5) Use softmax to compute loss of the predicted distribution.           #
        #                                                                          #
        # For simplicity, there is no one-hot sparse vector to dense vector        #
        # embedding, and only the RNN layer is followed by an activation           #
        # function, i.e. tanh. Activation should be implemented in the             #
        # rnn_step_forward and rnn_step_backward functions, not here.              #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        
        # Forward prop
        rnn_h, rnn_cache = rnn_forward(wordvecs, h0, Wx, Wh, b)
        # print("h.shape: {0}".format(h.shape))
        # print("W_a.shape: {0}".format(W_a.shape))
        ta_out, ta_cache = temporal_affine_forward(rnn_h, W_a, b_a)
        # print("ta_out.shape: {0}".format(ta_out.shape))
        # A = ta_out.shape[2]
        # print("A: {0}".format(A))
        av_out, av_cache = average_forward(ta_out, mask)
        # print("av_out.shape: {0}".format(av_out.shape))
        # print("mask.shape: {0}".format(mask.shape))
        # print(av_out)
        # print(mask)
        
        a_out, a_cache = affine_forward(av_out, W_fc, b_fc)
        # print("W_fc.shape: {0}".format(W_fc.shape))
        # print("a_out.shape: {0}".format(a_out.shape))
        
        loss, grads = softmax_loss(a_out, labels)
        
        # Back prop
        # print("grads.shape: {0}".format(grads.shape))
        
        a_dx, a_dw, a_db = affine_backward(grads, a_cache)
        
        # print("a_dx.shape: {0}".format(a_dx.shape))
        # print("a_dw.shape: {0}".format(a_dw.shape))
        # print("a_db.shape: {0}".format(a_db.shape))
        
        av_dhi = average_backward(a_dx, av_cache)
        
        # print("av_dhi.shape: {0}".format(av_dhi.shape))
        
        ta_dx, ta_dw, ta_db = temporal_affine_backward(av_dhi, ta_cache)
        
        # print("ta_dx.shape: {0}".format(ta_dx.shape))
        # print("ta_dw.shape: {0}".format(ta_dw.shape))
        # print("ta_db.shape: {0}".format(ta_db.shape))
        
        dx, dh0, dWx, dWh, db = rnn_backward(ta_dx, rnn_cache)
        
        # print("grads.shape: {0}".format(grads.shape))
        
        grads = {
            'Wx': dWx, 'Wh': dWh, 'b': db,
            'W_a': ta_dw, 'b_a': ta_db,
            'W_fc': a_dw, 'b_fc': a_db
        }
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def inference(self, wordvecs, mask):
        """
        Run a test-time forward pass for the model. We input word vectors, ground-truth 
        labels for sentence sentiments and masks for the lengths of sentences, and use 
        an RNN to generate sentiment class probability distribution w.r.t. input word 
        vectors.

        Inputs:
        - wordvecs: Array of input word vectors of shape (N, T, V).
        - mask: indices of sentence length, of shape (N, T). Each sentence has a max
          length of T with 0 padding at the end if its actual length is smaller than T.

        Returns:
        - preds: Array of shape (N, 2) giving the predicted probability distribution,
          where each element is in the range of [0, 1.].
        """
        preds = None

        # Unpack parameters
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_a, b_a = self.params['W_a'], self.params['b_a']
        W_fc, b_fc = self.params['W_fc'], self.params['b_fc']

        N, T, V = wordvecs.shape
        H = self.params['Wh'].shape[0]
        h0 = np.zeros((N, H))
    
        ############################################################################
        # TODO: Implement test-time sampling for the model. At each timestep you   #
        #  will need to do the following for only the forward pass:                #
        # (1) Use a vanilla RNN to process the sequence of input word vectors      #
        #     and produce an array of hidden states at all timesteps, of shape     #
        #     (N, T, H).                                                           #
        # (2) Use the temporal affine tranformation to map the concatenated        #
        #     array into a new space, producing an array of shape (N, T, A).       #
        # (3) Use the average vector of all timesteps as a hidden representation   #
        #     of each sentence, producing an output array of shape (N, A). Ignore  #
        #     the points where the length is beyond the actual sentence length     # 
        #     using the mask above.                                                #   
        # (4) Apply an affine transformation to map it into a 2-D vector of shape  #
        #     (N, 2).                                                              #
        # (5) Use softmax to produce the sentiment probability distribution,       #
        #     product an output array of shape (N, 2).                             #
        #                                                                          #
        # For simplicity, there is no one-hot sparse vector to dense vector        #
        # embedding, and only the RNN layer is followed by an activation           #
        # function, i.e. tanh. Activation should be implemented in the             #
        # rnn_step_forward and rnn_step_backward functions, not here.              #
        #                                                                          #
        ############################################################################
        
        # Forward prop
        rnn_h, rnn_cache = rnn_forward(wordvecs, h0, Wx, Wh, b)
        ta_out, ta_cache = temporal_affine_forward(rnn_h, W_a, b_a)
        av_out, av_cache = average_forward(ta_out, mask)
        a_out, a_cache = affine_forward(av_out, W_fc, b_fc)
        preds = softmax(a_out)
        print("preds.shape: {0}".format(preds.shape))
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return preds
