# making a neural network from scratch with n layers

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore



# a neural network is a series of layers that are connected to each other and it contains neurons that have weights and biases that are updated during the training process

class NeuralNetwork:
    
    def __init__(self, inputs, hidden_layers_units:int, outputs, epochs:int, learning_rate:float):
        self.inputs = inputs
        self.hidden_layers_units = hidden_layers_units
        self.outputs = outputs
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = []
        self.bias = []
        
        # initialize the hidden weights and biases, 
        # using random values and dimensions which are the number of features in the input data and the number of hidden units
       
        self.hidden_weights = np.random.randn(inputs.shape[1], self.hidden_layers_units)
        self.hidden_biases = np.random.randn(1, self.hidden_layers_units)
        
       
        # output weights and biases
        self.output_weights = np.random.randn(self.hidden_layers_units, outputs.shape[1])
        self.output_biases = np.random.randn(1, outputs.shape[1])
        
        
        

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x*(1-x)
    
    
    # forward propagation which is the process of moving the input data through the network to get the output
    # multiply the input data by the weights and add the biases from the hidden layers to the output layer
    def forward_propagation(self, inputs): 
      
        print("input shape: ", inputs.shape)
        print("hidden weights shape: ", self.hidden_weights.shape)
        print("hidden biases shape: ", self.hidden_biases.shape)
        
        weighted_sum = np.dot(inputs, self.hidden_weights) + self.hidden_biases
        hidden_output = self.sigmoid(weighted_sum)
        
        output_layer_wsum = np.dot(hidden_output, self.output_weights) + self.output_biases
        output = self.sigmoid(output_layer_wsum)
    
        return output

    # mse loss function
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    # back propagation is the process of updating the weights and biases of the network to minimize the loss
    def back_propagation(self):
        pass
    
    # this function is used to train the Neural Network
    def train (self):
        pass
    

    
    # feed forward propagation


    
        
    
    
    

# what do we need to do?
# 1. Initialize the parameters
# 2. Forward Propagation
# 3. Compute the cost
# 4. Backward Propagation
# 5. Update the parameters
# 6. Repeat 2-5 until convergence


# main function
__name__ == "__main__"

# xor dataset
inputs = np.array([[1, 1],
  [1, 0],
  [0, 1],
  [0, 0]])
outputs = np.array([[0], [1], [1], [0]])

# create a neural network
nn = NeuralNetwork(inputs, 2, outputs, 1000, 0.1)
nn.forward_propagation(inputs)








