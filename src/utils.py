import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import numpy as np

from activations import LinearActivation, ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation


def plot_weights_distribution(model, layers_indices):
    for idx in layers_indices:
        if idx < len(model.layers):
            plt.figure()
            plt.hist(model.layers[idx].W.flatten(), bins=30)
            plt.title(f"Distribusi Bobot - Layer {idx}")
            plt.xlabel("Nilai Bobot")
            plt.ylabel("Frekuensi")
            plt.show()


def plot_gradients_distribution(model, layers_indices):
    for idx in layers_indices:
        if idx < len(model.layers):
            plt.figure()
            plt.hist(model.layers[idx].dW.flatten(), bins=30)
            plt.title(f"Distribusi Gradien - Layer {idx}")
            plt.xlabel("Nilai Gradien")
            plt.ylabel("Frekuensi")
            plt.show()
    
def visualize_network(model):    
    n_layers = len(model.layers) + 1
    layer_names = ['Input Layer'] + [f'Layer {i+1}' for i in range(len(model.layers))]

    _, ax = plt.subplots(figsize=(12, 8))

    layer_xs = np.linspace(0.1, 0.9, n_layers)
    neuron_radius = 0.015

    for idx in range(n_layers):
        if idx == 0:
            neurons = model.layers[0].input_dim
        else:
            neurons = model.layers[idx - 1].output_dim
        neuron_ys = np.linspace(0.1, 0.9, neurons)

        for i, y in enumerate(neuron_ys):
            if idx == 0:
                color = 'skyblue'
            else:
                activation = model.layers[idx-1].activation
                if isinstance(activation, SigmoidActivation):
                    color = 'salmon'
                elif isinstance(activation, ReLUActivation):
                    color = 'lightgreen'
                elif isinstance(activation, TanhActivation):
                    color = 'orange'
                elif isinstance(activation, SoftmaxActivation):
                    color = 'purple'
                elif isinstance(activation, LinearActivation):
                    color = 'gray'
                else:
                    color = 'lightgray'

            circle = plt.Circle((layer_xs[idx], y), neuron_radius, color=color, fill=True, ec='black')
            ax.add_patch(circle)

            if idx > 0:
                prev_layer_neurons = model.layers[idx - 1].input_dim if idx==1 else model.layers[idx-2].output_dim
                prev_ys = np.linspace(0.1, 0.9, prev_layer_neurons)
                for j, prev_y in enumerate(prev_ys):
                    weight = model.layers[idx-1].W[j, i]
                    lw = 0.5 + 3 * abs(weight) / (abs(model.layers[idx-1].W).max() + 1e-10)
                    conn_color = 'green' if weight > 0 else 'red'
                    ax.plot([layer_xs[idx-1], layer_xs[idx]], [prev_y, y], color=conn_color, linewidth=lw, alpha=0.6)

        if idx == 0:
            neurons_str = str(model.layers[0].input_dim)
            ax.text(layer_xs[idx], 0.02, f'{layer_names[idx]}\n{neurons_str} neurons', ha='center', va='bottom', fontsize=9)

        else:
             act_name = type(model.layers[idx-1].activation).__name__.replace('Activation', '')
             ax.text(layer_xs[idx], 0.02, f'{layer_names[idx]}\n{model.layers[idx-1].output_dim} neurons\n{act_name}', ha='center', va='bottom',fontsize=9)

    legend_elements = [
        patches.Patch(facecolor='skyblue', edgecolor='black', label='Input Neuron'),
        patches.Patch(facecolor='salmon', edgecolor='black', label='Sigmoid'),
        patches.Patch(facecolor='lightgreen', edgecolor='black', label='ReLU'),
        patches.Patch(facecolor='orange', edgecolor='black', label='Tanh'),
        patches.Patch(facecolor='purple', edgecolor='black', label='Softmax'),
        patches.Patch(facecolor='gray', edgecolor='black', label='Linear')
    ]

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('Neural Network Architecture', fontsize=12)
    plt.tight_layout()
    plt.show()

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model