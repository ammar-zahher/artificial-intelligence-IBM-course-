import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
d = np.array([[0, 1, 1, 0]])


def initialize_network_parameters():
    input_size = 2
    hidden_size = 2
    output_size = 1
    lr = 0.1
    epochs = 100000
    w1 = np.random.rand(hidden_size, input_size) * 2 - 1
    w2 = np.random.rand(output_size, hidden_size) * 2 - 1
    b1 = np.random.rand(hidden_size, 1) * 2 - 1
    b2 = np.random.rand(output_size, 1) * 2 - 1
    return w1, b1, w2, b2, lr, epochs


w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

list_error = []

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(w1, X) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # Backpropagation
    error = d - a2
    dz2 = error * (a2 * (1 - a2))

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (a1 * (1 - a1))

    # Update parameters
    w2 += lr * np.dot(dz2, a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)
    w1 += lr * np.dot(dz1, X.T)
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)

    if (epoch + 1) % 10000 == 0:
        avg_error = np.average(np.abs(error))
        print(f"Epoch: {epoch+1}, Average error: {avg_error:.05f}")
        list_error.append(avg_error)

print("\nFinal output after training:\n", a2)
print("Ground truth:", d)

plt.plot(list_error)
plt.title("Error Reduction")
plt.xlabel("Steps (x10000)")
plt.ylabel("Error")
plt.show()
