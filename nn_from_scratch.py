import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    print("Loading training data...")
    train_data = pd.read_csv(train_csv_path)

    print("Loading test data...")
    test_data = pd.read_csv(test_csv_path)

    y_train_full = train_data.iloc[:, 0].values
    X_train_full = train_data.iloc[:, 1:].values / 255.0

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0

    n_val = int(len(X_train_full) * val_split)
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
)


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        return grad_output


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        relu_grad = (input > 0).astype(float)
        return grad_output * relu_grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]
        grad_biases = np.mean(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)
    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    return np.argmax(logits, axis=-1)


def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1]
    logits = layer_activations[-1]

    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    grad_output = grad_logits
    for i in range(len(network))[::-1]:
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)

    return loss


def train_mnist_network(X_train, y_train, X_val, y_val, num_epochs=5, batch_size=32, subset_size=10000):
    network = [
        Dense(784, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 10),
    ]

    print("Network architecture:")
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            print(f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})")
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")

    train_loss_history = []
    val_accuracy_history = []

    if subset_size and subset_size < len(X_train):
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    print(f"\nTraining on {len(X_train_subset)} examples")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        indices = np.random.permutation(len(X_train_subset))
        X_shuffled = X_train_subset[indices]
        y_shuffled = y_train_subset[indices]

        num_batches = (len(X_train_subset) + batch_size - 1) // batch_size
        epoch_losses = []

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(X_train_subset))

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            loss = train(network, X_batch, y_batch)
            epoch_losses.append(loss)
            train_loss_history.append(loss)

            if batch % 20 == 0:
                print(f"  Batch {batch+1}/{num_batches}, Loss: {loss:.4f}")

        val_subset_size = min(1000, len(X_val))
        val_indices = np.random.choice(len(X_val), val_subset_size, replace=False)
        val_predictions = predict(network, X_val[val_indices])
        val_accuracy = np.mean(val_predictions == y_val[val_indices])
        val_accuracy_history.append(val_accuracy)

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch complete - Avg Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return network, val_accuracy


network, val_accuracy = train_mnist_network(X_train, y_train, X_val, y_val)
assert val_accuracy > 0.65

