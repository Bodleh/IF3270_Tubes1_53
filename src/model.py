import numpy as np        
import matplotlib.pyplot as plt

from activations import ELUActivation, LeakyReLUActivation, LinearActivation, ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation
from losses import CategoricalCrossEntropy, MSELoss, BinaryCrossEntropy
from initializers import initialize_weights
from utils import visualize_network


class Layer:
    def __init__(self, input_dim, output_dim, activation, weight_init_method="random_uniform", init_params=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W, self.b = initialize_weights(
            input_dim, output_dim, method=weight_init_method, init_params=init_params)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
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
        self.layers: list[Layer] = []
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
            elif act_name.lower() == "leakyrelu":
                alpha = 0.01
                if len(layers_config[i]) > 2:
                    alpha = layers_config[i][2]
                activation = LeakyReLUActivation(alpha=alpha)
            elif act_name.lower() == "elu":
                alpha = 1.0
                if len(layers_config[i]) > 2:
                    alpha = layers_config[i][2]
                activation = ELUActivation(alpha=alpha)
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
        if isinstance(self.layers[-1].activation, SoftmaxActivation) and isinstance(self.loss_func, CategoricalCrossEntropy):
            output_grad = y_pred - y_true
            gradient = output_grad
            
            for layer in reversed(self.layers):
                if layer == self.layers[-1]:
                    dZ = gradient
                    layer.dW = np.dot(layer.X.T, dZ) / layer.X.shape[0]
                    layer.db = np.sum(dZ, axis=0, keepdims=True) / layer.X.shape[0]
                    gradient = np.dot(dZ, layer.W.T)
                else:
                    gradient = layer.backward(gradient)
        else:
            gradient = self.loss_func.gradient(y_true, y_pred)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.01, verbose=1):
        
        # Using tqdm for better training visualization (included in requirements.txt) 
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x: x
                    
        history = {"train_loss": [], "val_loss": []}
        n_samples = X_train.shape[0]
        
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=verbose==0)
        for epoch in epoch_iterator:
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            epoch_loss = 0
            n_batches = int(np.ceil(n_samples / batch_size))
            
            batch_iterator = tqdm(range(0, n_samples, batch_size), 
                                desc=f"Epoch {epoch+1}/{epochs}", 
                                leave=False,
                                disable=verbose==0)
            
            for i in batch_iterator:
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                batch_loss = self.loss_func.loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                self.backward(y_batch, y_pred)
                self.update(learning_rate)

                if verbose:
                    batch_iterator.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
            
            epoch_loss /= n_batches
            history["train_loss"].append(epoch_loss)

            val_loss_str = ""
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_func.loss(y_val, y_val_pred)
                history["val_loss"].append(val_loss)
                val_loss_str = f", val_loss: {val_loss:.4f}"

            if verbose:
                epoch_iterator.set_postfix({"train_loss": f"{epoch_loss:.4f}{val_loss_str}"})

        if verbose and epochs > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label='Training Loss')
            if "val_loss" in history and history["val_loss"]:
                plt.plot(history["val_loss"], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
                
        return history

    def summary(self):
        print("FFNN Model's Layer:")
        for idx, layer in enumerate(self.layers):
            print(
                f"Layer {idx}: Input Dim = {layer.input_dim}, Output Dim = {layer.output_dim}, Activation = {type(layer.activation).__name__}")
            print(f"  Bobot: {layer.W.shape}, Bias: {layer.b.shape}")

    def visualize(self):
        visualize_network(self)