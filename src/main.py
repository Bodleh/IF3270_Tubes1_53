import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.model import FFNN
from src.losses import CategoricalCrossEntropy
from src.utils import save_model


def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32) / 255.0  # Normalisasi ke rentang [0, 1]
    y = mnist.target.astype(np.int32)
    # One-hot encoding untuk label
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    return X, y_onehot


def main():
    X, y = load_mnist()

    # ( 80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Format :(jumlah_neuron, nama_fungsi_aktivasi)
    layers_config = [
        (784, "linear"),    # Input layer (aktivasi linear sebagai placeholder)
        (128, "relu"),      # Hidden layer 1
        (64, "relu"),       # Hidden layer 2
        (10, "softmax")     # Output layer
    ]

    # loss bisa berupa string (pilihan: "mse", "binary_crossentropy", "categorical_crossentropy")
    # weight_init_method: "zero", "random_uniform", "random_normal"
    model = FFNN(layers_config, loss="categorical_crossentropy",
                 weight_init_method="random_uniform",
                 init_params={"lower": -0.05, "upper": 0.05, "seed": 42})

    model.summary()

    # Training model dengan parameter yang diinginkan
    epochs = 10
    batch_size = 64
    learning_rate = 0.01
    verbose = 1
    history = model.train(X_train, y_train, X_val=X_test, y_val=y_test,
                          epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, verbose=verbose)

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training dan Validation Loss")
    plt.legend()
    plt.show()

    save_model(model, "ffnn_model.pkl")
    print("Model telah disimpan ke ffnn_model.pkl")


if __name__ == '__main__':
    main()
