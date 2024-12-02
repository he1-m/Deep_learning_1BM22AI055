import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weights, bias, activation_function):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_function = activation_function

    def activate(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

weights = [0.5, -0.6]
bias = 0.1
inputs = np.array([[0.1, 0.2], [0.5, 0.8], [-0.3, 0.7], [0.8, -0.1]])

activation_functions = {'Sigmoid': sigmoid, 'Tanh': tanh, 'ReLU': relu}

results = {}
for name, activation in activation_functions.items():
    neuron = Neuron(weights, bias, activation)
    outputs = np.array([neuron.activate(x) for x in inputs])
    results[name] = outputs
    print(f"{name} Activation Outputs: \n{outputs}\n")

x_range = np.linspace(-2, 2, 100)
plt.figure(figsize=(10, 6))
plt.plot(x_range, sigmoid(x_range), label='Sigmoid')
plt.plot(x_range, tanh(x_range), label='Tanh')
plt.plot(x_range, relu(x_range), label='ReLU')
plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
