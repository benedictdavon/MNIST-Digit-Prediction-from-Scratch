import numpy as np

class NN():
    def __init__(self, n_input, n_hidden, n_output, lr, epochs):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # xavier initialization
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(1 / n_input)
        # small constant
        self.b1 = np.full((n_hidden, 1), 0.01)

        self.W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1 / n_hidden)
        self.b2 = np.full((n_output, 1), 0.01)

        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def loss(self, y, A2):
        m = y.shape[1]
        L = -1 * (1/m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        return L

    def forward(self, x):
        Z1 = self.W1 @ x + self.b1
        A1 = self.tanh(Z1)

        Z2 = self.W2 @ A1 + self.b2
        A2 = self.sigmoid(Z2)
        
        self.Z1, self.A1 = Z1, A1
        self.Z2, self.A2 = Z2, A2

        return self.A2

    def backward(self, x, Y):
        m = x.shape[1]

        # Output layer
        dZ2 = self.A2 - Y
        dW2 = (1/m) * dZ2 @ self.A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * (1 - self.A1**2)  # derivative of tanh
        dW1 = (1/m) * dZ1 @ x.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Save gradients
        self.dW1, self.db1, self.dW2, self.db2 = dW1, db1, dW2, db2

    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def train(self, x, Y):
        for epoch in range(self.epochs):
            A2 = self.forward(x)

            loss = self.loss(Y, A2)

            self.backward(x, Y)

            self.update()

            if epoch % (self.epochs/10) == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X, threshold = 0.5):
        A2 = self.forward(X)            # probabilities
        predictions = (A2 >= threshold).astype(int)
        return predictions
