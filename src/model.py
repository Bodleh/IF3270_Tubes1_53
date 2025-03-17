import numpy as np
from .activations import LinearActivation, ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation
from .losses import CategoricalCrossEntropy, MSELoss, BinaryCrossEntropy
from .initializers import initialize_weights


class Layer:
    def __init__(self, input_dim, output_dim, activation, weight_init_method="random_uniform", init_params=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # Inisialisasi bobot dan bias menggunakan
        self.W, self.b = initialize_weights(
            input_dim, output_dim, method=weight_init_method, init_params=init_params)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X  # Menyimpan input untuk backpropagation
        self.Z = np.dot(X, self.W) + self.b
        self.A = self.activation.forward(self.Z)
        return self.A

    def backward(self, dA):
        dZ = dA * self.activation.backward(self.Z)
        self.dW = np.dot(self.X.T, dZ) / self.X.shape[0]
        self.db = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]
        dX = np.dot(dZ, self.W.T)
        return dX

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class FFNN:
    def __init__(self, layers_config, loss, weight_init_method="random_uniform", init_params=None):
        self.layers = []
        if isinstance(loss, str):
            loss = loss.lower()
            if loss == "mse":
                self.loss_func = MSELoss()
            elif loss == "binary_crossentropy":
                self.loss_func = BinaryCrossEntropy()
            elif loss == "categorical_crossentropy":
                self.loss_func = CategoricalCrossEntropy()
            else:
                raise ValueError(
                    "Loss function string tidak dikenali: " + loss)
        else:
            self.loss_func = loss
        for i in range(1, len(layers_config)):
            input_dim = layers_config[i-1][0]
            output_dim = layers_config[i][0]
            act_name = layers_config[i][1]
            if act_name.lower() == "linear":
                activation = LinearActivation()
            elif act_name.lower() == "relu":
                activation = ReLUActivation()
            elif act_name.lower() == "sigmoid":
                activation = SigmoidActivation()
            elif act_name.lower() == "tanh":
                activation = TanhActivation()
            elif act_name.lower() == "softmax":
                activation = SoftmaxActivation()
            else:
                raise ValueError("Unknown activation function: " + act_name)
            layer = Layer(input_dim, output_dim, activation,
                          weight_init_method, init_params)
            self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        grad = self.loss_func.gradient(y_true, y_pred)
        # Backpropagation dimulai dari layer terakhir
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.01, verbose=1):
        history = {"train_loss": [], "val_loss": []}
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                y_pred = self.forward(X_batch)
                loss = self.loss_func.loss(y_batch, y_pred)
                epoch_loss += loss
                self.backward(y_batch, y_pred)
                self.update(learning_rate)
            epoch_loss /= (n_samples / batch_size)
            history["train_loss"].append(epoch_loss)
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_func.loss(y_val, y_val_pred)
                history["val_loss"].append(val_loss)
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
        return history

    def summary(self):
        print("Ringkasan Model FFNN:")
        for idx, layer in enumerate(self.layers):
            print(
                f"Layer {idx}: Input Dim = {layer.input_dim}, Output Dim = {layer.output_dim}, Activation = {type(layer.activation).__name__}")
            print(f"  Bobot: {layer.W.shape}, Bias: {layer.b.shape}")
