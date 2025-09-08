import numpy as np
from nn import NN

def generate_xor_data(n):
    combos = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    X = combos.T  # shape (n, 2^n)
    Y = (np.sum(combos, axis=1) % 2).reshape(1, -1)  # odd parity = 1
    return X, Y
    
def test(n, n_input, n_hidden, lr, epochs):
    x, Y = generate_xor_data(n)

    model = NN(n_input=n_input, n_hidden=n_hidden, n_output=1, lr=lr, epochs=epochs)

    model.train(x, Y)

    preds = model.predict(x)

    print(f"Predicting XOR for {n} inputs:")
    print(f"XOR-{n} Predictions:", preds)
    print(f"Accuracy: {np.mean(preds == Y) * 100:.2f}%")

if __name__ == "__main__":
    # Example usage
    np.random.seed(0)

    test(n=2, n_input=2, n_hidden=4, lr=0.1, epochs=10_000)
    test(n=3, n_input=3, n_hidden=8, lr=0.2, epochs=10_000)
    test(n=4, n_input=4, n_hidden=16, lr=0.05, epochs=10_000)
    test(n=5, n_input=5, n_hidden=32, lr=0.05, epochs=10_000)

    

