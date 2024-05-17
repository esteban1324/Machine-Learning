# making a neural network from scratch with n layers

#using wsl so TkAgg is needed.
import numpy as np # type: ignore
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore



# a neural network is a series of layers that are connected to each other and it contains neurons that have weights and biases that are updated during the training process

class NeuralNetwork:
    
    def __init__(self, input_layer_size, hidden_layers_sizes, outputs_layer_size, epochs:int, learning_rate:float):
        self.input_layer_size = input_layer_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.outputs_layer_size = outputs_layer_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = []
        self.bias = []
        self.losses = []
        
        # initialize the weights and biases for all the layers
        layer_sizes = [self.input_layer_size] + self.hidden_layers_sizes + [self.outputs_layer_size]
        # we have NN list with units for each layer, initialize the weights and biases with each unit.
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.bias.append(np.random.rand(1,layer_sizes[i+1]))
            
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x*(1-x)
    
    # forward propagation which is the process of moving the input data through the network to get the output
    # multiply the input data by the weights and add the biases from the hidden layers to the output layer
    def forward_propagation(self, inputs): 
        activated_inputs = [inputs]

        for i in range(len(self.weights)):
            weighted_sum = np.dot(activated_inputs[-1], self.weights[i]) + self.bias[i]
            activation = self.sigmoid(weighted_sum)
            activated_inputs.append(activation)
        
        return activated_inputs

    # mse loss function
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    # back propagation is the process of updating the weights and biases of the network to minimize the loss
    def back_propagation(self, activated_inputs, y_true):
        # get error btw the true outcome and activations 
        errors = [y_true - activated_inputs[-1]]

        # compute gradient from right to left with general formula 
        for i in reversed(range(len(self.weights) - 1)):
            delta = errors[-1].dot(self.weights[i+1].T) * self.sigmoid_derivative(activated_inputs[i+1])
            errors.append(delta)
        
        errors.reverse()
        
        # update weights and biases with errors 
        for i in range(len(self.weights)):
            self.weights[i] += activated_inputs[i].T.dot(errors[i]) * self.learning_rate
            self.bias[i] += np.sum(errors[i], axis=0) * self.learning_rate
            
            
    
    # this function is used to train the Neural Network
    def train (self, X, y):
        
        for i in range(self.epochs):
            activations =  self.forward_propagation(X)
            self.back_propagation(activations, y)
            loss = self.loss(y, activations[-1])
            self.losses.append(loss)
    
    def predict(self, X):
        return self.forward_propagation(X)[-1]

# plotting functions for NN

#plot learning rate 
def plot_learning_rate(nn):
    plt.plot(range(nn.epochs), nn.losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("learning rate")
    plt.show()

# plot loss rate
def plot_decision_boundary(nn):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1), np.arange(y_min, y_max, 0.1))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], s=40, edgecolor='k')
    plt.show()


# xor dataset
X = np.array([[1, 1],
    [1, 0],
    [0, 1],
    [0, 0]])
y = np.array([[0], [1], [1], [0]])

# main function
if __name__ == "__main__":

    # create a neural network
    nn = NeuralNetwork(2, [2], 1, 2000, 0.1)
    nn.train(X, y)
    plot_learning_rate(nn)
    plot_decision_boundary(nn)







