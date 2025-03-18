import matplotlib.pyplot as plt
import pickle
import numpy as np

def visualize_network(model):
    print("Visualize network using graph with weights and gradients info")

def plot_weights_distribution(model, layers_indices: list[int]):
    n_layers = len(layers_indices)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for i, idx in enumerate(layers_indices):
        if idx < len(model.layers):
            weights = model.layers[idx].W.flatten()
            
            axs[i].hist(weights, bins=30, color=f'C{i}', edgecolor='black', linewidth=0.5)
            axs[i].set_title(f"Layer {idx} Weights")
            axs[i].grid(True, linestyle='--', alpha=0.3)
    
    for i in range(len(layers_indices), len(axs)):
        axs[i].set_visible(False)
    
    fig.text(0.5, 0.02, 'Weight Value', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)
    
    fig.suptitle("Weight Distributions", fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show()

def plot_gradients_distribution(model, layers_indices: list[int]):
    n_layers = len(layers_indices)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    for i, idx in enumerate(layers_indices):
        if idx < len(model.layers):
            gradients = model.layers[idx].dW.flatten()
            
            axs[i].hist(gradients, bins=30, color=f'C{i}', edgecolor='black', linewidth=0.5)
            axs[i].set_title(f"Layer {idx} Gradients")
            axs[i].grid(True, linestyle='--', alpha=0.3)
    
    for i in range(len(layers_indices), len(axs)):
        axs[i].set_visible(False)
    
    fig.text(0.5, 0.02, 'Gradient Value', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)
    
    fig.suptitle("Gradient Distributions", fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show()

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model