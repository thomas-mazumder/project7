# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess

# TODO: Write your test functions and associated docstrings below.

def initialize_network(loss):
    """
    Helper function to initialize a network with non-random weights
    """
    layers = [{'input_dim': 1, 'output_dim': 1, 'activation': 'relu' },]
    net = nn.NeuralNetwork(layers, lr = 0.1, seed = 0, batch_size = 1, epochs = 1, 
                         loss_function = loss)
    net._param_dict['W1'] = np.array([[1]])
    net._param_dict['b1'] = np.array([[1]])
    return net

def data():
    """
    Helper function to create a small data set
    """
    X = np.array([[1]])
    y = np.array([1])
    return X, y

def test_forward():
    """
    Test that forward() returns the correct output and cache
    """
    net = initialize_network("MSE")
    X, y = data()
    output, cache = net.forward(X)
    assert np.all(np.isclose(output, np.array([[2]])))
    assert np.all(np.isclose(cache['0'], np.array([[1]])))
    assert np.all(np.isclose(cache['1'][0], np.array([[2]])))
    assert np.all(np.isclose(cache['1'][0], np.array([[2]]))) 

def test_single_forward():
    """
    Test that _single_forward() returns the correct linear output
    and activation-transformed output
    """
    net = initialize_network("MSE")
    X, y = data()
    W_curr = net._param_dict['W1']
    b_curr = net._param_dict['b1']
    A, Z = net._single_forward(W_curr, b_curr, X, "relu")
    assert np.all(np.isclose(A, np.array([[2]])))
    assert np.all(np.isclose(Z, np.array([[2]])))

def test_single_backprop():
    """
    Test that _single_backprop() returns the correct gradients
    """
    net = initialize_network("MSE")
    X, y = data()
    W_curr = net._param_dict['W1']
    b_curr = net._param_dict['b1']
    A, Z = net._single_forward(W_curr, b_curr, X, "relu")
    dJ_dA = net._mean_squared_error_backprop(y, A)
    dJ_dWcurr, dJ_dbcurr, dJ_dAprev = net._single_backprop(W_curr, b_curr, Z, 
                                                          X, dJ_dA, "relu")
    assert np.all(np.isclose(dJ_dWcurr, np.array([[2]])))
    assert np.all(np.isclose(dJ_dbcurr, np.array([[2]])))
    assert np.all(np.isclose(dJ_dAprev, np.array([[2]])))
    pass


def test_predict():
    """
    Test that predict() returns the correct output
    """
    net = initialize_network("MSE")
    X, y = data()
    output = net.predict(X)
    assert np.all(np.isclose(output, np.array([[2]])))

def test_binary_cross_entropy():
    """
    Test that _binary_cross_entropy() returns the correct loss
    """
    net = initialize_network("BCE")
    y = np.array([[1.]])
    y_hat = np.array([[0.5]])
    bce = net._binary_cross_entropy(y, y_hat)
    assert np.isclose(bce, -np.log(.5))

def test_binary_cross_entropy_backprop():
    """
    Test that _binary_cross_entropy_backprop() returns the correct gradient
    """
    net = initialize_network("BCE")
    y = np.array([[0.]])
    y_hat = np.array([[0.5]])
    grad = net._binary_cross_entropy_backprop(y, y_hat)
    assert np.all(np.isclose(grad, np.array([[2]])))

def test_mean_squared_error():
    """
    Test that _binary_cross_entropy() returns the correct loss
    """
    net = initialize_network("MSE")
    y = np.array([[1.]])
    y_hat = np.array([[1.]])
    mse = net._binary_cross_entropy(y, y_hat)
    assert np.isclose(mse, 0, atol = 0.001)

def test_mean_squared_error_backprop():
    """
    Test that _mean_squared_error_backprop() returns the correct gradient
    """
    net = initialize_network("MSE")
    y = np.array([[1.]])
    y_hat = np.array([[0.]])
    grad = net._mean_squared_error_backprop(y, y_hat)
    assert np.all(np.isclose(grad, np.array([[-2]])))

def test_one_hot_encode():
    seq_arr = ["ATCG"]
    target = np.array([[1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1]])
    result = preprocess.one_hot_encode_seqs(seq_arr)
    assert np.all(np.isclose(target, result))

def test_sample_seqs():
    seqs = ['a','b','c']
    labels = [1, 1, 0]
    target_seqs = ['a','b','c','c']
    target_labels = [1, 1, 0, 0]
    result_seqs, result_labels = preprocess.sample_seqs(seqs, labels)
    assert target_seqs == result_seqs
    assert target_labels == result_labels
