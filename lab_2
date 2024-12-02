import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def activation_function(self, z):
        # Step function
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """
        Train the perceptron.
        :param X: Input features, shape (n_samples, n_features)
        :param y: Target labels (0 or 1), shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)
                
                # Perceptron learning rule
                error = y[idx] - y_pred
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

    def predict(self, X):
        """
        Predict labels for given inputs.
        :param X: Input features, shape (n_samples, n_features)
        :return: Predicted labels (0 or 1)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)


# Test the perceptron on a simple binary classification task
if __name__ == "__main__":
    # OR gate dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])  # OR gate output

    perceptron = Perceptron(learning_rate=0.1, max_iter=1000)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)

    print("Predictions:", predictions)
    print("Weights:", perceptron.weights)
    print("Bias:", perceptron.bias)
