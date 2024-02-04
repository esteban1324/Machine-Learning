import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("C:\\Users\\esteb\\PythonProjects\\projects\\Machine-Learning\\Perceptron\\pima-indians-diabetes.csv", header=None, names=column_names)

X = df.drop(['Outcome'], axis=1).to_numpy()
y = df['Outcome']

# Question 1.2 
# Plot the decision boundary for this perceptron using the points in the previous question
#  (0,0), (0,1), (1,0), (1,1), (0,0.5), (0.5,0), (0.5,0.5)
# w1 = 1 w2 = 1, b = -.4
def plot_decision_boundary():
    # plot the decision boundary
    
    # plot the points above  
    plt.scatter(0, 0, c='b')
    plt.scatter(0, 1, c='b')
    plt.scatter(1, 0, c='b')
    plt.scatter(1, 1, c='b')
    plt.scatter(0, 0.5, c='b')
    plt.scatter(0.5, 0, c='b')
    plt.scatter(0.5, 0.5, c='b')
    
    # plot the decision boundary
    x = np.linspace(-1, 2)
    
    plt.xlim(-.2, 1.2)
    plt.ylim(-.2, 1.2)
    
    y = -x + 0.4
    plt.plot(x, y, '-r', label='y=-x+0.4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    
    plt.show()

    
# Question 3.1

# plot an XOR graph, where the points are colored based on their class
def plot_xor():
    
    # plot the points above and color them based on their class
    plt.scatter(0, 0, c='b')
    plt.scatter(0, 1, c='r')
    plt.scatter(1, 0, c='r')
    plt.scatter(1, 1, c='b')
    
    # limit the x and y axis
    plt.xlim(-.1, 1.1)
    plt.ylim(-.1, 1.1)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Perceptron class which that will be built and used to train the model
class Perceptron:
    
    # Initialize the perceptron with weights, bias, learning rate, and epochs
    def __init__(self, weights, bias, learning_rate, max_itr):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.max_itr = max_itr
        self.num_errors = []
        self.rate = []
      
     
    # Activation function and y = w^T * x + b
    def input(self, X):
        return np.dot(X, self.weights) + self.bias 
    
    # threshold function that returns 1 if the input is greater than 0 and 0 otherwise
    def predict(self, X):
        return np.where(self.input(X) > 0.0, 1.0, 0.0)
    
    # perceptron learning algorithm
    def fit(self, X, y):   
        # for each epoch  
        for i in range(self.max_itr):
            # for each training example in the dataset
            error = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                
                if output != target:
                    error += 1
                    self.weights += self.learning_rate *  (target - output) * xi
                    self.bias += self.learning_rate * (target - output)   
                          
            self.num_errors.append(error)
            self.rate.append(error / len(X))
            
            
        return self    
    
        
        
    
# function that expirements with different learning rates.
def expirement():
    
    weights = [rd.random() for i in range (8)]
    print(weights)
    bias = 0.2
    max_itr = 150

    learning_rates = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, .2, .4]

    # vars best learning rate and best accuracy score
    best_learning_rate = 0
    best_accuracy = 0

    # get the best accuracy score using the different learning rates
    for learning_rate in learning_rates:
        
        perceptron = Perceptron(weights, bias, learning_rate, max_itr).fit(X,y)
        num_error = perceptron.num_errors
        error_rate = perceptron.rate
        
        # get the best accuracy score
        if 1-min(error_rate) > best_accuracy:
            best_accuracy = 1-min(error_rate)
            best_learning_rate = learning_rate
          
        # plots classifciation errors vs # of epochs.
        x_range = np.arange(max_itr)
        plt.xlabel(f"# Epochs with learning rate: {learning_rate:.5f}")
        plt.ylabel("# of errors ")
        plt.plot(x_range, num_error, marker = 'o', markersize = 5, linewidth = 2)
        plt.show()
 
    print("best learning rate: ", best_learning_rate)
    print("best accuracy score: ", best_accuracy)
 
# Main function
if __name__ == "__main__":
    expirement()